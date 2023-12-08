import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import set_seed, accuracy, MultipleOptimizer

class NodeClassification_GAug(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, seed, beta, warmup, pretrain_ep, pretrain_nc):
        super(NodeClassification_GAug, self).__init__()

        self.__dataset = dataset
        self.__model = model 

        self.__lr = lr 
        self.__weight_decay = weight_decay

        self.__epochs = epochs 
        self.__device = device 
        self.__seed = seed

        self.__warmup = warmup
        self.__beta = beta

        self.__pretrain_ep = pretrain_ep
        self.__pretrain_nc = pretrain_nc

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
    def col_normalization(features):
        """ column normalization for feature matrix """
        features = features.numpy()
        m = features.mean(axis=0)
        s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
        features -= m
        features /= s
        return torch.FloatTensor(features)
    
    def pretrain_ep_net(self, model, adj, features, adj_orig, norm_w, pos_weight):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(model.ep_net.parameters(),
                                     lr=self.__lr)
        model.train()
        for _ in range(self.__pretrain_ep):
            adj_logits = model.ep_net(adj, features)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            if not model.gae:
                mu = model.ep_net.mean
                lgstd = model.ep_net.logstd
                kl_divergence = 0.5 / adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                loss -= kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def pretrain_nc_net(self, model, adj, features, labels):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(model.nc_net.parameters(),
                                     lr=self.__lr,
                                     weight_decay=self.__weight_decay)
        # loss function for node classification
        if labels.dim() == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.
        for _ in range(self.__pretrain_nc):
            model.train()
            nc_logits = model.nc_net(features, adj)
            # losses
            loss = nc_criterion(nc_logits[self.__dataset.train_idx], labels[self.__dataset.train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(features, adj)
            val_acc = accuracy(nc_logits_eval[self.__dataset.val_idx], labels[self.__dataset.val_idx])
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    def _execute(self):
        set_seed(self.__seed)

        features, adj_orig, adj_norm, adj = self.__model.preprocess(self.__dataset.x, self.__dataset.adj, self.__device)
        
        model = self.__model.to(self.__device)
        labels = self.__dataset.y.to(self.__device)

        # weights for log_lik loss when training EP net
        adj_t = adj_orig
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.__device)
        # pretrain VGAE if needed
        if self.__pretrain_ep:
            self.pretrain_ep_net(model, adj_norm, features, adj_orig, norm_w, pos_weight)
        # pretrain GCN if needed
        if self.__pretrain_nc:
            self.pretrain_nc_net(model, adj, features, labels)
        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(model.ep_net.parameters(),
                                                    lr=self.__lr),
                                   torch.optim.Adam(model.nc_net.parameters(),
                                                    lr=self.__lr,
                                                    weight_decay=self.__weight_decay))
        # get the learning rate schedule for the optimizer of ep_net if needed
        if self.__warmup:
            ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.__epochs, self.__lr, self.__warmup)
        # loss function for node classification
        if labels.dim() == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        
        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        patience_step = 0
        # train model
        for epoch in range(self.__epochs):
            # update the learning rate for ep_net if needed
            if self.__warmup:
                optims.update_lr(0, ep_lr_schedule[epoch])

            model.train()
            nc_logits, adj_logits = model(adj_norm, adj_orig, features)

            # losses
            loss = nc_criterion(nc_logits[self.__dataset.train_idx], labels[self.__dataset.train_idx])
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            loss += self.__beta * ep_loss
            optims.zero_grad()
            loss.backward()
            optims.step()
            # validate (without dropout)
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(features, adj)
            val_acc = accuracy(nc_logits_eval[self.__dataset.val_idx], labels[self.__dataset.val_idx])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = accuracy(nc_logits_eval[self.__dataset.test_idx], labels[self.__dataset.test_idx])
                patience_step = 0
            else:
                patience_step += 1
                if patience_step == 100:
                    break
        # release RAM and GPU memory
        del adj, features, labels, adj_orig
        torch.cuda.empty_cache()
        gc.collect()
        
        return test_acc