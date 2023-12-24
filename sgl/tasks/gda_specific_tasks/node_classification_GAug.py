import gc
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import set_seed, accuracy, MultipleOptimizer

class NodeClassificationGAugO(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, beta, warmup, max_patience, pretrain_ep, pretrain_nc, runs=1, verbose=True, seed=12345, pretrain_batch_size=None, train_batch_size=None, ep_lr=None):
        super(NodeClassificationGAugO, self).__init__()

        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = MultipleOptimizer(Adam(model.ep_net.parameters(), lr=lr),
                                             Adam(model.nc_net.parameters(), lr=lr, weight_decay=weight_decay))

        self.__lr = lr 
        self.__ep_lr = ep_lr if ep_lr is not None else lr
        self.__weight_decay = weight_decay

        self.__epochs = epochs 
        self.__device = device 
        self.__seed = seed
        self.__runs = runs 
        self.__verbose = verbose

        self.__warmup = warmup
        self.__beta = beta
        self.__max_patience = max_patience

        self.__pretrain_ep = pretrain_ep
        self.__pretrain_nc = pretrain_nc
        self.__pretrain_batch_size = pretrain_batch_size
        self.__train_batch_size = train_batch_size

        self.__test_acc = self._execute()

    @property
    def test_acc(self):
        return self.__test_acc

    @staticmethod
    def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
        """ schedule the learning rate with the sigmoid function.
        The learning rate will start with near zero and end with near lr """
        factors = torch.FloatTensor(np.arange(n_epochs))
        factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
        factors = torch.sigmoid(factors)
        # range the factors to [0, 1]
        factors = (factors - factors[0]) / (factors[-1] - factors[0])
        lr_schedule = factors * lr
        return lr_schedule

    @staticmethod
    def loss_fn(nc_logits, norm_w, adj_logits, adj_orig, pos_weight, labels, global_idx, beta, local_idx=None):
        if labels.dim() == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        if local_idx is None:
            local_idx = global_idx
        loss = nc_criterion(nc_logits[local_idx], labels[global_idx])
        ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
        loss += beta * ep_loss

        return loss
    
    @staticmethod
    def extend_batch(seed_batch, hops, adj_matrix):
        nodes_batch = seed_batch
        for _ in range(hops):
            neigh_block = adj_matrix[nodes_batch]
            nodes_batch = neigh_block.sum(0).nonzero()[1]
        nodes_batch = np.setdiff1d(nodes_batch, seed_batch, assume_unique=True)
        nodes_batch = np.concatenate((seed_batch, nodes_batch))
        return nodes_batch
    
    def _pretrain_ep_net(self, adj, features, adj_orig, norm_w, pos_weight):
        """ pretrain the edge prediction network """
        optimizer = Adam(self.__model.ep_net.parameters(), lr=self.__ep_lr)

        self.__model.train()
        for _ in range(self.__pretrain_ep):
            adj_logits = self.__model.ep_net(adj, features)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            if not self.__model.gae:
                mu = self.__model.ep_net.mean
                lgstd = self.__model.ep_net.logstd
                kl_divergence = 0.5 / adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                loss -= kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def _minibatch_pretrain_ep_net(self, adj, features, adj_orig, norm_w, pos_weight):
        """ pretrain the edge prediction network in mini-batches"""
        optimizer = Adam(self.__model.ep_net.parameters(), lr=self.__ep_lr)
        num_nodes = features.size(0)
        train_loader = DataLoader(range(num_nodes), batch_size=self.__pretrain_batch_size, shuffle=True, drop_last=False)

        self.__model.train()
        for _ in range(self.__pretrain_ep):
            for node_batch in train_loader:
                sub_adj_orig = adj_orig[node_batch][:, node_batch].to(self.__device)
                sub_adj_logits = self.__model.ep_net(adj, features, node_batch)
                loss = norm_w * F.binary_cross_entropy_with_logits(sub_adj_logits, sub_adj_orig, pos_weight=pos_weight)
                if not self.__model.gae:
                    mu = self.__model.ep_net.mean
                    lgstd = self.__model.ep_net.logstd 
                    kl_divergence = 0.5 / sub_adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                    loss -= kl_divergence
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _pretrain_nc_net(self, adj, features):
        """ pretrain the node classification network """
        optimizer = Adam(self.__model.nc_net.parameters(), lr=self.__lr, weight_decay=self.__weight_decay)
        # loss function for node classification
        if self.__labels.dim() == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        for _ in range(self.__pretrain_nc):
            self.__model.train()
            nc_logits = self.__model.nc_net(features, adj)
            # losses
            loss = nc_criterion(nc_logits[self.__dataset.train_idx], self.__labels[self.__dataset.train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _train(self, adj_norm, adj_orig, features, norm_w, pos_weight, epoch, ep_lr_schedule):
        # update the learning rate for ep_net if needed
        if self.__warmup:
            self.__optimizer.update_lr(0, ep_lr_schedule[epoch])

        self.__model.train()
        nc_logits, adj_logits = self.__model(adj_norm, adj_orig, features)
        loss_train = self.loss_fn(nc_logits, norm_w, adj_logits, adj_orig, pos_weight, self.__labels, self.__dataset.train_idx, self.__beta)
        acc_train = accuracy(nc_logits[self.__dataset.train_idx], self.__labels[self.__dataset.train_idx])
        self.__optimizer.zero_grad()
        loss_train.backward()
        self.__optimizer.step()

        return loss_train, acc_train

    def _minibatch_train(self, adj_matrix, adj_norm, adj_orig, features, norm_w, pos_weight, epoch, ep_lr_schedule):
        # update the learning rate for ep_net if needed
        if self.__warmup:
            self.__optimizer.update_lr(0, ep_lr_schedule[epoch])
        
        seed_size = self.__train_batch_size // 20
        num_batches = int((len(self.__dataset.train_idx) + seed_size - 1) / seed_size)
        node_idx_all = np.array(self.__dataset.train_idx)
        np.random.shuffle(node_idx_all)
        seed_batches = np.array_split(node_idx_all, num_batches)

        train_loss = 0.
        train_num, num_correct = 0, 0
        self.__model.train()
        for seed_batch in seed_batches:
            nodes_batch = self.extend_batch(seed_batch, 2, adj_matrix)
            if len(nodes_batch) >= self.__train_batch_size:
                nodes_batch = nodes_batch[:self.__train_batch_size]

            sub_adj_orig = adj_orig[nodes_batch][:, nodes_batch].to(self.__device)
            nc_logits, sub_adj_logits = self.__model(adj_norm, sub_adj_orig, features, nodes_batch)
            num_correct += nc_logits[range(len(seed_batch))].argmax(dim=1).eq(self.__labels[seed_batch]).sum().long().item()
            loss = self.loss_fn(nc_logits, norm_w, sub_adj_logits, sub_adj_orig, pos_weight, self.__labels, seed_batch, self.__beta, range(len(seed_batch)))
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            train_loss += loss.item() * len(seed_batch)
            train_num += len(seed_batch)
        
        return train_loss / train_num, num_correct / train_num  

    def _evaluate(self, features, adj):
        self.__model.eval()
        with torch.no_grad():
            nc_logits_eval = self.__model.nc_net(features, adj)
        acc_val = accuracy(nc_logits_eval[self.__dataset.val_idx], self.__labels[self.__dataset.val_idx])
        acc_test = accuracy(nc_logits_eval[self.__dataset.test_idx], self.__labels[self.__dataset.test_idx])

        return acc_val, acc_test

    def _execute(self):
        set_seed(self.__seed)
        
        features, adj_orig, adj, adj_norm = self.__model.preprocess(self.__dataset.x, self.__dataset.adj, self.__device)
        if self.__train_batch_size is not None:
            adj_matrix = sp.csr_matrix(adj_orig.numpy())

        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        # weights for log_lik loss when training EP net
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_orig.shape[0]**2 - adj_orig.sum()) / adj_orig.sum()]).to(self.__device)
        
        acc_test_list = []
        for _ in range(self.__runs):
            # reset model parameters at the beginning of each run
            self.__model.reset_parameters()
            # pretrain VGAE if needed
            if self.__pretrain_ep:
                if self.__pretrain_batch_size is None:
                    self._pretrain_ep_net(adj_norm, features, adj_orig, norm_w, pos_weight)
                else:
                    self._minibatch_pretrain_ep_net(adj_norm, features, adj_orig, norm_w, pos_weight)
            # pretrain GCN if needed
            if self.__pretrain_nc:
                self._pretrain_nc_net(adj, features)
            # get the learning rate schedule for the optimizer of ep_net if needed
            if self.__warmup:
                ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.__epochs, self.__lr, self.__warmup)
            else:
                ep_lr_schedule = None

            # keep record of the best validation accuracy for early stopping
            best_acc_val, best_acc_test, patience_step = 0., 0., 0
            # train model
            for epoch in range(self.__epochs):
                t = time.time()
                
                if self.__train_batch_size is None:
                    loss_train, acc_train = self._train(adj_norm, adj_orig, features, norm_w, pos_weight, epoch, ep_lr_schedule)
                else:
                    loss_train, acc_train = self._minibatch_train(adj_matrix, adj_norm, adj_orig, features, norm_w, pos_weight, epoch, ep_lr_schedule)
                acc_val, acc_test = self._evaluate(features, adj)

                if self.__verbose:
                    print('Epoch: {:03d}'.format(epoch + 1),
                        'loss_train: {:.4f}'.format(loss_train),
                        'acc_train: {:.4f}'.format(acc_train),
                        'acc_val: {:.4f}'.format(acc_val),
                        'acc_test: {:.4f}'.format(acc_test),
                        'time: {:.4f}s'.format(time.time() - t))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test
                    patience_step = 0
                else:
                    patience_step += 1
                    if patience_step == self.__max_patience:
                        break
            
            acc_test_list.append(best_acc_test)
        
        # release RAM and GPU memory
        del adj, features, adj_orig, adj_norm
        torch.cuda.empty_cache()
        gc.collect()
        
        return np.mean(acc_test_list)


class NodeClassificationGAugM(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, runs=1, verbose=True, loss_fn=nn.CrossEntropyLoss(), seed=42, max_patience=100):
        super(NodeClassificationGAugM, self).__init__()

        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed
        self.__max_patience = max_patience
        self.__runs = runs 
        self.__verbose = verbose

        self.__test_acc = self._execute()

    @property
    def test_acc(self):
        return self.__test_acc

    def _train(self, adj_norm, features):
        self.__model.train()
        pred_y = self.__model(adj_norm, features)[self.__dataset.train_idx]
        ground_truth_y = self.__labels[self.__dataset.train_idx]
        loss_train = self.__loss_fn(pred_y, ground_truth_y)
        acc_train = accuracy(pred_y, ground_truth_y)

        self.__optimizer.zero_grad()
        loss_train.backward()
        self.__optimizer.step()

        return loss_train, acc_train

    def _evaluate(self, adj_norm, features):
        self.__model.eval()
        with torch.no_grad():
            pred_y = self.__model(adj_norm, features)
        acc_val = accuracy(pred_y[self.__dataset.val_idx], self.__labels[self.__dataset.val_idx])
        acc_test = accuracy(pred_y[self.__dataset.test_idx], self.__labels[self.__dataset.test_idx])

        return acc_val, acc_test

    def _execute(self):
        set_seed(self.__seed)

        pre_time_st = time.time()
        adj_pred_dir = os.path.join(self.__dataset.processed_dir, "GAugM_edge_probabilities")
        adj, features = self.__model.preprocess(self.__dataset.adj, self.__dataset.x, adj_pred_dir, self.__device)
        pre_time_ed = time.time()
        if self.__verbose:
            print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")

        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        acc_val_list = []
        acc_test_list = []

        for _ in range(self.__runs):
            self.__model.reset_parameters()
            t_total = time.time()
            best_val = 0.
            best_test = 0.
            patience = 0
            for epoch in range(self.__epochs):
                t = time.time()
                loss_train, acc_train = self._train(adj, features)
                acc_val, acc_test = self._evaluate(adj, features)

                if self.__verbose:
                    print('Epoch: {:03d}'.format(epoch + 1),
                        'loss_train: {:.4f}'.format(loss_train),
                        'acc_train: {:.4f}'.format(acc_train),
                        'acc_val: {:.4f}'.format(acc_val),
                        'acc_test: {:.4f}'.format(acc_test),
                        'time: {:.4f}s'.format(time.time() - t))

                if acc_val > best_val:
                    best_val = acc_val
                    best_test = acc_test
                    patience = 0
                else:
                    patience += 1
                    if patience == self.__max_patience:
                        break

            acc_val_list.append(best_val)
            acc_test_list.append(best_test)

        mean_acc_test = np.mean(acc_test_list)    
        if self.__verbose:      
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            print(f'Best val: {np.mean(acc_val_list):.4f}, best test: {mean_acc_test:.4f}')

        del adj, features
        torch.cuda.empty_cache()
        gc.collect()

        return mean_acc_test

