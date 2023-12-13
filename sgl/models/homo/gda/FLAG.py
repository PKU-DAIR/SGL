import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from sgl.models.simple_models import GCN, SAGE, GAT
from sgl.operators.graph_op import LaplacianGraphOp, RwGraphOp
from sgl.utils import sparse_mx_to_torch_sparse_tensor

class FLAG(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, step_size, augM, activation=F.relu, **kwargs):
        super(FLAG, self).__init__()
        self.__step_size = step_size 
        self.__augM = augM
        self.__gnn_type = gnn_type
        gnn_backbone = {"gcn": GCN, "sage": SAGE, "gat": GAT}
        if gnn_type == 'gat':
            if kwargs.get("n_heads"):
                n_heads = list(map(lambda x: int(x), kwargs["n_heads"].split(",")))
            else:
                n_heads = [8] * (n_layers - 1) + [1]
            kwargs.update({"n_heads": n_heads})
            activation = F.elu
        self.nc_net = gnn_backbone.get(gnn_type)(in_dim, hidden_dim, n_classes, n_layers=n_layers, dropout=dropout, activation=activation, **kwargs)

    @property
    def processed_feature(self):
        return self.__features
    
    @property
    def processed_adj(self):
        return self.__processed_adj
    
    def preprocess(self, adj, features, device):
        self.__features = features.to(device)
        if self.__gnn_type == "gcn":
            adj_norm = LaplacianGraphOp()._construct_adj(adj)
            self.__processed_adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
        elif self.__gnn_type == "sage":
            adj_norm = RwGraphOp()._construct_adj(adj)
            self.__processed_adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
        elif self.__gnn_type == "gat":
            adj_sl = sp.coo_matrix(adj)
            adj_sl = adj_sl + sp.eye(*adj_sl.shape)
            self.__processed_adj = torch.FloatTensor(adj_sl.todense()).to(device)

    def flag(self, ground_truth_y, optimizer, device, train_idx, loss_fn):
        x = self.__features
        adj = self.__processed_adj
        
        self.nc_net.train()
        optimizer.zero_grad()

        perturb = torch.FloatTensor(x.shape).uniform_(-self.__step_size, self.__step_size).to(device)
        perturb.requires_grad_()
        pred_y = self.nc_net(x+perturb, adj)[train_idx]
        loss = loss_fn(pred_y, ground_truth_y)
        loss /= self.__augM

        for _ in range(self.__augM-1):
            loss.backward()
            perturb_data = perturb.detach() + self.__step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0 

            pred_y = self.nc_net(x+perturb, adj)[train_idx]
            loss = loss_fn(pred_y, ground_truth_y)
            loss /= self.__augM
        
        loss.backward()
        optimizer.step()

        return loss

    def train_func(self, train_idx, labels, device, optimizer, loss_fn, metric):
        loss_train = self.flag(labels[train_idx], optimizer, device, train_idx, loss_fn)

        self.nc_net.eval()
        pred_y = self.nc_net(self.__features, self.__processed_adj)
        acc_train = metric(pred_y[train_idx], labels[train_idx])

        return loss_train.item(), acc_train
    
    @torch.no_grad()
    def evaluate_func(self, val_idx, test_idx, labels, device, metric):
        self.nc_net.eval()
        pred_y = self.nc_net(self.__features, self.__processed_adj)

        acc_val = metric(pred_y[val_idx], labels[val_idx])
        acc_test = metric(pred_y[test_idx], labels[test_idx])
        return acc_val, acc_test
    
    def model_forward(self, idx, device):
        pred_y = self.forward(self.__features, self.__processed_adj)
        return pred_y[idx]

    def forward(self, x, adj):
        return self.nc_net(x, adj)
    
    def postprocess(self, adj, outputs):
        return outputs