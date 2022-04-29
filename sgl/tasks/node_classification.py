import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import accuracy, set_seed, train, mini_batch_train, evaluate, mini_batch_evaluate


class NodeClassification(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(), seed=42,
                 train_batch_size=None, eval_batch_size=None):
        super(NodeClassification, self).__init__()

        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        self.__mini_batch = False
        if train_batch_size is not None:
            self.__mini_batch = True
            self.__train_loader = DataLoader(
                self.__dataset.train_idx, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.__val_loader = DataLoader(
                self.__dataset.val_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.__test_loader = DataLoader(
                self.__dataset.test_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.__all_eval_loader = DataLoader(
                range(self.__dataset.data.num_node), batch_size=eval_batch_size, shuffle=False, drop_last=False)

        self.__test_acc = self._execute()

    @property
    def test_acc(self):
        return self.__test_acc

    def _execute(self):
        set_seed(self.__seed)

        pre_time_st = time.time()
        self.__model.preprocess(self.__dataset.adj, self.__dataset.x)
        pre_time_ed = time.time()
        print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")

        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        for epoch in range(self.__epochs):
            t = time.time()
            if self.__mini_batch is False:
                loss_train, acc_train = train(self.__model, self.__dataset.train_idx, self.__labels, self.__device,
                                              self.__optimizer, self.__loss_fn)
                acc_val, acc_test = evaluate(self.__model, self.__dataset.val_idx, self.__dataset.test_idx,
                                             self.__labels, self.__device)
            else:
                loss_train, acc_train = mini_batch_train(self.__model, self.__dataset.train_idx, self.__train_loader,
                                                         self.__labels, self.__device, self.__optimizer, self.__loss_fn)
                acc_val, acc_test = mini_batch_evaluate(self.__model, self.__dataset.val_idx, self.__val_loader,
                                                        self.__dataset.test_idx, self.__test_loader, self.__labels,
                                                        self.__device)

            print('Epoch: {:03d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'acc_val: {:.4f}'.format(acc_val),
                  'acc_test: {:.4f}'.format(acc_test),
                  'time: {:.4f}s'.format(time.time() - t))
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test

        acc_val, acc_test = self._postprocess()
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        return best_test

    def _postprocess(self):
        self.__model.eval()
        if self.__mini_batch is False:
            outputs = self.__model.model_forward(
                range(self.__dataset.num_node), self.__device).to("cpu")
        else:
            outputs = None
            for batch in self.__all_eval_loader:
                output = self.__model.model_forward(batch, self.__device)
                if outputs is None:
                    outputs = output
                else:
                    outputs = torch.vstack((outputs, output))

        final_output = self.__model.postprocess(self.__dataset.adj, outputs)
        acc_val = accuracy(
            final_output[self.__dataset.val_idx], self.__labels[self.__dataset.val_idx])
        acc_test = accuracy(
            final_output[self.__dataset.test_idx], self.__labels[self.__dataset.test_idx])
        return acc_val, acc_test


class HeteroNodeClassification(BaseTask):
    def __init__(self, dataset, predict_class, model, lr, weight_decay, epochs,
                 device, loss_fn=nn.CrossEntropyLoss(), seed=42,
                 train_batch_size=None, eval_batch_size=None,
                 random_subgraph_num=-1, subgraph_edge_type_num=-1,
                 subgraph_list=None, record_subgraph_weight=False):

        if subgraph_list is None and (random_subgraph_num == -1 or subgraph_edge_type_num == -1):
            raise ValueError(
                "Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided!")
        if subgraph_list is not None and (random_subgraph_num != -1 or subgraph_edge_type_num != -1):
            raise ValueError(
                "subgraph_list is provided, random_subgraph_num and subgraph_edge_type_num will be ignored!")

        super(HeteroNodeClassification, self).__init__()

        self.__dataset = dataset
        self.__predict_class = predict_class
        self.__labels = self.__dataset.data[self.__predict_class].y.squeeze(-1)

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        self.__mini_batch = False
        if train_batch_size is not None:
            self.__mini_batch = True
            self.__train_loader = DataLoader(
                self.__dataset.train_idx, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.__val_loader = DataLoader(
                self.__dataset.val_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.__test_loader = DataLoader(
                self.__dataset.test_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)

        self.__test_acc, self.__subgraph_weight = self._execute(
            random_subgraph_num, subgraph_edge_type_num,
            subgraph_list, record_subgraph_weight)

    @property
    def test_acc(self):
        return self.__test_acc

    @property
    def subgraph_weight(self):
        return self.__subgraph_weight

    def _execute(self, random_subgraph_num=-1, subgraph_edge_type_num=-1,
                 subgraph_list=None, record_subgraph_weight=False):
        set_seed(self.__seed)

        pre_time_st = time.time()
        self.__model.preprocess(
            self.__dataset, self.__predict_class,
            random_subgraph_num, subgraph_edge_type_num, subgraph_list)
        pre_time_ed = time.time()
        print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")

        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        if record_subgraph_weight:
            best_subgraph_weight = self.__model.subgraph_weight.clone()
        for epoch in range(self.__epochs):
            t = time.time()
            if self.__mini_batch is False:
                loss_train, acc_train = train(self.__model, self.__dataset.train_idx, self.__labels, self.__device,
                                              self.__optimizer, self.__loss_fn)
                acc_val, acc_test = evaluate(self.__model, self.__dataset.val_idx, self.__dataset.test_idx,
                                             self.__labels, self.__device)
            else:
                loss_train, acc_train = mini_batch_train(self.__model, self.__dataset.train_idx, self.__train_loader,
                                                         self.__labels, self.__device, self.__optimizer, self.__loss_fn)
                acc_val, acc_test = mini_batch_evaluate(self.__model, self.__dataset.val_idx, self.__val_loader,
                                                        self.__dataset.test_idx, self.__test_loader, self.__labels,
                                                        self.__device)

            print('Epoch: {:03d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'acc_val: {:.4f}'.format(acc_val),
                  'acc_test: {:.4f}'.format(acc_test),
                  'time: {:.4f}s'.format(time.time() - t))
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                if record_subgraph_weight:
                    best_subgraph_weight = self.__model.subgraph_weight.clone()

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')

        if record_subgraph_weight:
            return best_test, best_subgraph_weight
        else:
            return best_test, None
