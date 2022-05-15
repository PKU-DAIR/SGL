import time
from matplotlib.pyplot import sca
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import accuracy, set_seed, train, mini_batch_train, evaluate, \
                            mini_batch_evaluate, label_prop, adj_to_symmetric_norm


class CorrectAndSmooth(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, num_correct_layers, 
                 correct_alpha, num_smooth_layers, smooth_alpha, autoscale=True, scale=1.0, 
                 loss_fn=nn.CrossEntropyLoss(), seed=42, train_batch_size=None, eval_batch_size=None, ):
        super(CorrectAndSmooth, self).__init__()

        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        self.__autoscale = autoscale
        self.__scale = scale
        self.__num_correct_layers = num_correct_layers
        self.__num_smooth_layers = num_smooth_layers
        self.__correct_alpha = correct_alpha
        self.__smooth_alpha = smooth_alpha

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
        print(f"After C&S, acc_val: {acc_val:.4f} acc_test: {acc_test:.4f}")
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

        DAD = adj_to_symmetric_norm(adj=self.__dataset.adj, r=0.5)
        DA = adj_to_symmetric_norm(adj=self.__dataset.adj, r=0)

        final_output = self._correct(outputs, self.__labels, self.__dataset.train_idx, DAD, 
                                    self.__num_correct_layers, self.__correct_alpha)
        final_output = self._smooth(final_output, self.__labels, self.__dataset.train_idx, DA,
                                    self.__num_smooth_layers, self.__smooth_alpha)
        acc_val = accuracy(
            final_output[self.__dataset.val_idx], self.__labels[self.__dataset.val_idx])
        acc_test = accuracy(
            final_output[self.__dataset.test_idx], self.__labels[self.__dataset.test_idx])
        return acc_val, acc_test

    # different from pyg implemetation, y_true here represents all the labels for convenience
    @torch.no_grad()
    def _correct(self, y_soft, y_true, mask, adj, num_layers, alpha):
        y_soft = y_soft.cpu()
        y_true = y_true.cpu()
        mask = torch.tensor(mask)
        if y_true.dtype == torch.long:
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)
        # y_true.to(self.__device)
        # print("y_soft shape: ",y_soft.shape)
        # print("y_true.shape: ",y_true.shape)
        error = torch.zeros_like(y_soft)
        # print("error.device:", error.device)
        # print("y_soft.device:", y_soft.device)
        # print("y_true.device:", y_true.device)


        error[mask] = y_true[mask] - y_soft[mask]
        num_true = mask.shape[0] if mask.dtype == torch.long else int(mask.sum())

        if self.__autoscale:
            smoothed_error = label_prop(error, adj, num_layers, alpha, post_process=lambda x:x.clamp_(-1., 1.))
            sigma = error[mask].abs().sum() / num_true
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            return y_soft + smoothed_error * scale
        
        else:
            def fix_input(x):
                x[mask] = error[mask]
                return x

            smoothed_error = label_prop(error, adj, num_layers, alpha, post_process=fix_input)
            return smoothed_error * self.__scale + y_soft

    @torch.no_grad()
    def _smooth(self, y_soft, y_true, mask, adj, num_layers, alpha):
        y_soft = y_soft.cpu()
        y_true = y_true.cpu()
        if y_true.dtype == torch.long:
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)
        
        y_soft[mask] = y_true[mask]

        smoothed_label = label_prop(y_soft, adj, num_layers, alpha)
        return smoothed_label
