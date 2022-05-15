import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import accuracy, set_seed, train, mini_batch_train, evaluate, \
                            mini_batch_evaluate, adj_to_symmetric_norm
from sgl.tricks import CorrectAndSmooth

class NodeClassification_With_CorrectAndSmooth(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, num_correct_layers, 
                 correct_alpha, num_smooth_layers, smooth_alpha, autoscale=True, scale=1.0, 
                 loss_fn=nn.CrossEntropyLoss(), seed=42, train_batch_size=None, eval_batch_size=None, 
                 correct_r=0.5, smooth_r=0.5):
        super(NodeClassification_With_CorrectAndSmooth, self).__init__()

        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        self.__post_trick = CorrectAndSmooth(num_correct_layers, correct_alpha, 
                                        num_smooth_layers, smooth_alpha, autoscale, scale)
        self.__smooth_r = smooth_r
        self.__correct_r = correct_r

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
        best_y_soft = None
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
                idx = torch.arange(0, self.__dataset.num_node)
                output = self.__model.model_forward(idx, self.__device).detach()
                best_y_soft = F.softmax(output, dim=1)
                best_val = acc_val
                best_test = acc_test

        acc_val, acc_test = self._postprocess(best_y_soft)
        
        print(f"After C&S, acc_val: {acc_val:.4f} acc_test: {acc_test:.4f}")
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        return best_test

    def _postprocess(self, y_soft):
        self.__model.eval()

        correct_adj = adj_to_symmetric_norm(adj=self.__dataset.adj, r=self.__correct_r)
        smooth_adj = adj_to_symmetric_norm(adj=self.__dataset.adj, r=self.__smooth_r)
        
        post = self.__post_trick
        final_output = post.correct(y_soft, self.__labels, self.__dataset.train_idx, correct_adj)
        final_output = post.smooth(final_output, self.__labels, self.__dataset.train_idx, smooth_adj)

        acc_val = accuracy(
            final_output[self.__dataset.val_idx], self.__labels[self.__dataset.val_idx])
        acc_test = accuracy(
            final_output[self.__dataset.test_idx], self.__labels[self.__dataset.test_idx])
        return acc_val, acc_test
