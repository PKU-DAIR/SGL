import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import argparse
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from torch_geometric.utils import negative_sampling, from_scipy_sparse_matrix

import sgl.dataset as Dataset
from sgl.tasks.utils import set_seed
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.utils import sparse_mx_to_torch_sparse_tensor
from utils import sparse_to_tuple, get_scores_gen_graphs

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True):
        super(GraphConv, self).__init__()
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial)

    def forward(self, adj, inputs):
        x = inputs @ self.weight
        x = adj @ x
        if self.activation:
            return F.elu(x)
        else:
            return x

class VGAE(nn.Module):
    def __init__(self, dim_in, dim_h, dim_z, gae):
        super(VGAE,self).__init__()
        self.dim_z = dim_z
        self.gae = gae
        self.base_gcn = GraphConv(dim_in, dim_h)
        self.gcn_mean = GraphConv(dim_h, dim_z, activation=False)
        self.gcn_logstd = GraphConv(dim_h, dim_z, activation=False)

    def encode(self, adj, X):
        hidden = self.base_gcn(adj, X)
        self.mean = self.gcn_mean(adj, hidden)
        if self.gae:
            return self.mean
        else:
            self.logstd = self.gcn_logstd(adj, hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
            return sampled_z

    def decode(self, Z):
        A_pred = Z @ Z.T
        return A_pred

    def forward(self, adj, X):
        Z = self.encode(adj, X)
        A_pred = self.decode(Z)
        return A_pred

def prepare_data(dataset, val_frac, test_frac, no_mask, norm_feat=True):
    adj_ori, features_orig = dataset.adj, dataset.x 
    if adj_ori.diagonal().sum() > 0:
        adj_ori = sp.coo_matrix(adj_ori)
        adj_ori.setdiag(0)
        adj_ori.eliminate_zeros()
        adj_ori = sp.csr_matrix(adj_ori)
    if isinstance(features_orig, torch.Tensor):
        features_orig = features_orig.numpy()
    features_orig = sp.csr_matrix(features_orig)
    if norm_feat:
        features_orig = normalize(features_orig, norm="l1", axis=1)
    adj_triu = sp.triu(adj_ori)
    edges = sparse_to_tuple(adj_triu)[0]
    num_val = int(np.floor(edges.shape[0] * val_frac))
    num_test = int(np.floor(edges.shape[0] * test_frac))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val+num_test)]
    val_edges = edges[val_edge_idx]
    test_edges = edges[test_edge_idx]
    if no_mask:
        train_edges = edges 
    else:
        train_edge_idx = all_edge_idx[num_val+num_test:]
        train_edges = edges[train_edge_idx]
    
    num_nodes = adj_ori.shape[0]
    test_edges_false = negative_sampling(from_scipy_sparse_matrix(adj_ori+sp.eye(adj_ori.shape[0]))[0], num_nodes, num_test)
    test_edges_false = test_edges_false.numpy()

    val_edges_false = negative_sampling(from_scipy_sparse_matrix(adj_ori+sp.eye(adj_ori.shape[0]))[0], num_nodes, num_val)
    val_edges_false = val_edges_false.numpy()

    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj_ori.shape)
    adj_train = adj_train + adj_train.T 
    adj_norm = LaplacianGraphOp()._construct_adj(adj_train)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_label)
    features = sparse_mx_to_torch_sparse_tensor(features_orig)

    return features, adj_ori, adj_train, adj_norm, adj_label, val_edges, val_edges_false, test_edges, test_edges_false

def train_model(data, model, lr, epochs, gae, device, verbose=False, criterion="roc"):
    features, _, adj_train, adj_norm, adj_label, val_edges, val_edges_false, test_edges, test_edges_false = data
    optimizer = Adam(model.parameters(), lr=lr)
    adj_t = adj_train
    norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(device)
    features = features.to(device)
    adj_norm = adj_norm.to(device)
    adj_label = adj_label.to_dense().to(device)
    best_val = 0
    best_state_dict = None
    model.train()
    for epoch in range(epochs):
        adj_pred = model(adj_norm, features)
        optimizer.zero_grad()
        loss = norm_w * F.binary_cross_entropy_with_logits(adj_pred, adj_label, pos_weight=pos_weight)
        if gae is False:
            kl_divergence = 0.5 / adj_pred.size(0) * (1 + 2 * model.logstd - model.mean**2 - torch.exp(2*model.logstd)).sum(1).mean()
            loss -= kl_divergence
        
        adj_pred = torch.sigmoid(adj_pred).detach().cpu()
        scores_val = get_scores_gen_graphs(val_edges, val_edges_false, adj_pred, adj_label)
        if verbose:
            print("Epoch{:3}: train_loss: {:.4f} recon_acc: {:.4f} val_roc: {:.4f} val_ap: {:.4f} val_f1: {:.4f}".format(
                    epoch+1, loss.item(), scores_val["acc"], scores_val["roc"], scores_val["ap"], scores_val["f1"]))
        if scores_val[criterion] > best_val:
            best_val = scores_val[criterion]
            best_state_dict = copy.deepcopy(model.state_dict())
            if verbose:
                scores_test = get_scores_gen_graphs(test_edges, test_edges_false, adj_pred, adj_label)
                print("test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
                    scores_test["roc"], scores_test["ap"], scores_test["f1"], scores_test["acc"]))
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_state_dict)
    return model

def graph_generate(dataset, model, lr, epochs, val_frac, test_frac, no_mask, num_gen_graphs, device, criterion, norm_feat=True, gae=True, verbose=False):
    data = prepare_data(dataset, val_frac, test_frac, no_mask, norm_feat)
    model = model.to(device)
    model = train_model(data, model, lr, epochs, gae, device, verbose, criterion)
    adj_ori = data[1]
    save_dir = os.path.join(dataset.processed_dir, "GAugM_edge_probabilities")
    if gae:
        save_path = os.path.join(save_dir, "0_gae.pkl")
    else:
        save_path = os.path.join(save_dir, "0.pkl")
    pkl.dump(adj_ori, open(save_path, "wb"))
    features = data[0].to(device)
    adj_norm = data[3].to(device)
    for i in range(num_gen_graphs):
        with torch.no_grad():
            adj_pred = model(adj_norm, features)
        adj_pred = torch.sigmoid(adj_pred).detach().cpu()
        adj_recon = adj_pred.numpy()
        np.fill_diagonal(adj_recon, 0)
        if gae:
            save_path = os.path.join(save_dir, f"{i+1}_logits_gae.pkl")
        else:
            save_path = os.path.join(save_dir, f"{i+1}_logits.pkl")
        pkl.dump(adj_recon, open(save_path, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate graphs for GAugM")
    parser.add_argument("--emb_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_gen_graphs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--dataset_classname", type=str, default="Planetoid")
    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--criterion", type=str, default="roc")
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--gae", action="store_true")
    parser.add_argument("--root", type=str, default="/home/ssq/test_data/")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    dataset = getattr(Dataset, args.dataset_classname)(root=args.root, name=args.dataset_name)
    model = VGAE(dataset.num_features, args.hidden_size, args.emb_size, args.gae)
    graph_generate(dataset, model, args.lr, args.epochs, args.val_frac, args.test_frac, args.no_mask, args.num_gen_graphs, device, args.criterion, True, args.gae, verbose=True)