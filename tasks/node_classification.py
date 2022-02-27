import time
from torch.optim import Adam
import torch.nn as nn

from tasks.base_task import BaseTask
from tasks.utils import accuracy, set_seed


class NodeClassification(BaseTask):

    # data = (labels, train_idx, val_idx, test_idx)
    def __init__(self, data, model, lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(), seed=42):
        super(NodeClassification, self).__init__()
        if len(data) != 4:
            raise ValueError("The fist parameter must be a tuple (labels, idx_train, idx_val, idx_test)!")

        (self.__labels, self.__train_idx, self.__val_idx, self.__test_idx) = data
        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        self.__call__()

    def __call__(self):
        set_seed(self.__seed)
        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        for epoch in range(self.__epochs):
            t = time.time()
            loss_train, acc_train = self._train()
            acc_val, acc_test = self._evaluate()
            print('Epoch: {:03d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'acc_val: {:.4f}'.format(acc_val),
                  'acc_test: {:.4f}'.format(acc_test),
                  'time: {:.4f}s'.format(time.time() - t))
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        return best_test

    def _evaluate(self):
        self.__model.eval()
        output = self.__model.train_model(self.__device)

        acc_val = accuracy(output[self.__val_idx], self.__labels[self.__val_idx])
        acc_test = accuracy(output[self.__test_idx], self.__labels[self.__test_idx])
        return acc_val, acc_test

    def _train(self):
        self.__model.train()
        self.__optimizer.zero_grad()

        output = self.__model.train_model(self.__device)
        loss_train = self.__loss_fn(output[self.__train_idx], self.__labels[self.__train_idx])
        acc_train = accuracy(output[self.__train_idx], self.__labels[self.__train_idx])
        loss_train.backward()
        self.__optimizer.step()

        return loss_train.item(), acc_train
