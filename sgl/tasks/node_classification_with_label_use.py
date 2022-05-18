import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import accuracy, set_seed, train, mini_batch_train, \
                            evaluate, mini_batch_evaluate, add_labels

# Node Classification with Label use and Label Reuse trick
# NOTE: When use this trick, the input_dim of model should be feature_dim + num_classes instead of feature_dim
class NodeClassificationWithLabelUse(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(), seed=42,
                 train_batch_size=None, eval_batch_size=None, label_reuse_batch_size=None,
                 mask_rate=0.5, use_labels=True, reuse_start_epoch=0, label_iters=0):
        super(NodeClassificationWithLabelUse, self).__init__()

        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed
        self.__mask_rate = mask_rate
        self.__use_labels = use_labels
        self.__reuse_start_epoch = reuse_start_epoch
        self.__label_iters = label_iters
        self.__label_reuse_batch_size = label_reuse_batch_size

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

        if self.__label_iters > 0 and self.__use_labels is False:
            raise ValueError("When using label reuse, it's essential to enable label use!")

        self.__test_acc = self._execute()

    @property
    def test_acc(self):
        return self.__test_acc

    def _execute(self):
        set_seed(self.__seed)

        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.

        features = self.__dataset.x
        for epoch in range(self.__epochs):
            # add label feature if use_label is True
            if self.__use_labels:
                train_idx = np.array(self.__dataset.train_idx)
                mask = np.random.rand(train_idx.shape[0]) < self.__mask_rate
                train_labels_idx = train_idx[mask]
                train_pred_idx = train_idx[~mask]
                features = add_labels(self.__dataset.x, self.__dataset.y, train_labels_idx, self.__dataset.num_classes)
            
            pre_time_st = time.time()
            self.__model.preprocess(self.__dataset.adj, features)
            pre_time_ed = time.time()
            print(f"Feature Propagate done in {(pre_time_ed - pre_time_st):.4f}s")

            # label reuse
            # small optimization: only utilize the predicted soft_labels in later epoches
            if self.__label_iters > 0 and epoch > self.__reuse_start_epoch:
                model = self.__model
                full_idx = torch.arange(self.__dataset.num_node, dtype=torch.long)
                val_idx = np.array(self.__dataset.val_idx)
                test_idx = np.array(self.__dataset.test_idx)
                unlabeled_idx = np.concatenate([train_pred_idx, val_idx, test_idx])
                for _ in range(self.__label_iters):
                    pred = []
                    if self.__label_reuse_batch_size is not None:
                        label_reuse_loader = DataLoader(full_idx, batch_size=self.__label_reuse_batch_size, shuffle=False, drop_last=False)
                        for batch in label_reuse_loader:
                            tmp = model.model_forward(batch, self.__device)
                            pred.append(tmp)
                        pred = torch.cat(pred)
                    else:
                        pred = model.model_forward(full_idx, self.__device)
                    pred = pred.detach().cpu()
                    torch.cuda.empty_cache()
                    features[unlabeled_idx, -self.__dataset.num_classes:] = F.softmax(pred[unlabeled_idx], dim=-1)
                    self.__model.preprocess(self.__dataset.adj, features)

            t = time.time()
            if self.__mini_batch is False:
                loss_train, acc_train = train(self.__model, train_pred_idx, self.__labels, self.__device,
                                              self.__optimizer, self.__loss_fn)
                acc_val, acc_test = evaluate(self.__model, self.__dataset.val_idx, self.__dataset.test_idx,
                                             self.__labels, self.__device)
            else:
                loss_train, acc_train = mini_batch_train(self.__model, train_pred_idx, self.__train_loader,
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
