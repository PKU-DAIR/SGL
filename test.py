import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.SGAP_models import SGC, SIGN, SSGC, GAMLP
from dataset.planetoid import Planetoid


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch, model, labels, idx_train, idx_val, idx_test, device):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model.train_model(device)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model.train_model(device)

    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:03d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val, acc_test


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

dataset = Planetoid("cora", "./", "official")
adj, features, labels = dataset.adj, dataset.x, dataset.y
idx_train, idx_val, idx_test = dataset.train_idx, dataset.val_idx, dataset.test_idx

model = GAMLP(prop_steps=12, feat_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=32, num_layers=2)
model.preprocess(adj, features)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
labels = labels.to(device)

t_total = time.time()
best_val = 0.
best_test = 0.
for epoch in range(200):
    acc_val, acc_test = train(epoch, model, labels, idx_train, idx_val, idx_test, device)
    if acc_val > best_val:
        best_val = acc_val
        best_test = acc_test
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
