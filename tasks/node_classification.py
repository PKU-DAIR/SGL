import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from tasks.base_task import BaseTask
from tasks.utils import accuracy, set_seed


class NodeClassification(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(), seed=42,
                 train_batch_size=None, eval_batch_size=None):
        super(NodeClassification, self).__init__()

        self.__dataset = dataset
        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

        self._execute()

    def _execute(self):
        set_seed(self.__seed)

        pre_time_st = time.time()
        self.__model.preprocess(self.__dataset.adj, self.__dataset.x)
        pre_time_ed = time.time()
        print(f"Preprocessing done in {(pre_time_ed-pre_time_st):.4f}s")

        self.__model = self.__model.to(self.__device)
        self.__dataset.y = self.__dataset.y.to(self.__device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        for epoch in range(self.__epochs):
            t = time.time()
            if self.__mini_batch is False:
                loss_train, acc_train = self._train()
                acc_val, acc_test = self._evaluate()
            else:
                loss_train, acc_train = self._mini_batch_train()
                acc_val, acc_test = self._mini_batch_evaluate()

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
            outputs = self.__model.model_forward(range(self.__dataset.num_node), self.__device).to("cpu")
        else:
            outputs = None
            for batch in self.__all_eval_loader:
                output = self.__model.model_forward(batch, self.__device)
                if outputs is None:
                    outputs = output
                else:
                    outputs = torch.vstack((outputs, output))

        final_output = self.__model.postprocess(outputs)
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

    def _mini_batch_evaluate(self):
        self.__model.eval()
        correct_num_val, correct_num_test = 0, 0
        for batch in self.__val_loader:
            val_output = self.__model.model_forward(batch, self.__device)
            pred = val_output.max(1)[1].type_as(self.__dataset.y)
            correct_num_val += pred.eq(self.__dataset.y[batch]).double().sum()
        for batch in self.__test_loader:
            test_output = self.__model.model_forward(batch, self.__device)
            pred = test_output.max(1)[1].type_as(self.__dataset.y)
            correct_num_test += pred.eq(self.__dataset.y[batch]).double().sum()

        return correct_num_val / len(self.__dataset.val_idx), correct_num_test / len(self.__dataset.test_idx)

    def _train(self):
        self.__model.train()
        self.__optimizer.zero_grad()

        train_output = self.__model.model_forward(self.__dataset.train_idx, self.__device)
        loss_train = self.__loss_fn(train_output, self.__dataset.y[self.__dataset.train_idx])
        acc_train = accuracy(train_output, self.__dataset.y[self.__dataset.train_idx])
        loss_train.backward()
        self.__optimizer.step()

        return loss_train.item(), acc_train

    def _mini_batch_train(self):
        self.__model.train()
        correct_num = 0
        loss_train_sum = 0.
        for batch in self.__train_loader:
            train_output = self.__model.model_forward(batch, self.__device)
            loss_train = self.__loss_fn(train_output, self.__dataset.y[batch])

            pred = train_output.max(1)[1].type_as(self.__dataset.y)
            correct_num += pred.eq(self.__dataset.y[batch]).double().sum()
            loss_train_sum += loss_train.item()

            self.__optimizer.zero_grad()
            loss_train.backward()
            self.__optimizer.step()

        return loss_train_sum / len(self.__train_loader), correct_num / len(self.__dataset.train_idx)
