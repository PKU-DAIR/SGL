import time
from torch.optim import Adam
import torch.nn as nn

from tasks.base_task import BaseTask
from tasks.utils import accuracy, set_seed


class NodeClassification(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(), seed=42):
        super(NodeClassification, self).__init__()

        self.__dataset = dataset
        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        self._execute()

    def _execute(self):
        set_seed(self.__seed)
        self.__model.preprocess(self.__dataset.adj, self.__dataset.x)
        self.__model = self.__model.to(self.__device)
        self.__dataset.y = self.__dataset.y.to(self.__device)

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
        output = self.__model.model_forward(range(self.__dataset.num_node), self.__device)
        final_output = self.__model.postprocess(output)

        acc_val = accuracy(final_output[self.__dataset.val_idx], self.__dataset.y[self.__dataset.val_idx])
        acc_test = accuracy(final_output[self.__dataset.test_idx], self.__dataset.y[self.__dataset.test_idx])
        return acc_val, acc_test

    def _evaluate(self):
        self.__model.eval()
        val_output = self.__model.model_forward(self.__dataset.val_idx, self.__device)
        test_output = self.__model.model_forward(self.__dataset.test_idx, self.__device)

        acc_val = accuracy(val_output, self.__dataset.y[self.__dataset.val_idx])
        acc_test = accuracy(test_output, self.__dataset.y[self.__dataset.test_idx])
        return acc_val, acc_test

    def _train(self):
        self.__model.train()
        self.__optimizer.zero_grad()

        train_output = self.__model.model_forward(self.__dataset.train_idx, self.__device)
        loss_train = self.__loss_fn(train_output, self.__dataset.y[self.__dataset.train_idx])
        acc_train = accuracy(train_output, self.__dataset.y[self.__dataset.train_idx])
        loss_train.backward()
        self.__optimizer.step()

        return loss_train.item(), acc_train
