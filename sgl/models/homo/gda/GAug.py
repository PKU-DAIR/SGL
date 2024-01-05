import os
import pyro
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import scipy.sparse as sp

from sgl.utils import sparse_mx_to_pyg_sparse_tensor
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.models.pyg_simple_models import GCNConv, GCN, SAGE, GAT
from sgl.models.homo.gda.utils import RoundNoGradient, CeilNoGradient


class GAugO(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_size, n_classes, n_layers, dropout, gnn_type,
                 activation=F.relu, temperature=1, gae=False, alpha=1, feat_norm="row", sample_type="add_sample", **kwargs):
        super(GAugO, self).__init__()
        self.__temperature = temperature
        self.__alpha = alpha
        self.__sample_type = sample_type
        self.__minibatch = kwargs.pop("minibatch", False)
        # edge prediction network
        self.__gae = gae
        self.__feat_norm = feat_norm
        self.ep_net = VGAE(in_dim, hidden_dim, emb_size, F.relu, gae=gae)
        # node classification network
        gnn_backbone = {"gcn": GCN, "gsage": SAGE, "gat": GAT}
        if isinstance(activation, str):
            activation = getattr(F, activation)
        if gnn_type == "gat":
            if kwargs.get("n_heads"):
                n_heads = list(map(lambda x: int(x), kwargs["n_heads"].split(",")))
            else:
                n_heads = [8] * (n_layers - 1) + [1]
            kwargs.update({"n_heads": n_heads})

        self.nc_net = gnn_backbone.get(gnn_type)(in_dim, hidden_dim, n_classes, n_layers=n_layers, dropout=dropout, activation=activation, **kwargs)
    
    @property
    def gae(self):
        return self.__gae
    
    @staticmethod
    def col_normalization(features):
        """ column normalization for feature matrix """
        features = features.numpy()
        m = features.mean(axis=0)
        s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
        features -= m
        features /= s
        return torch.FloatTensor(features)
    
    def reset_parameters(self):
        self.ep_net.reset_parameters()
        self.nc_net.reset_parameters()
        
    def preprocess(self, features, adj_matrix, device):
        if self.__feat_norm == "row":
            features = F.normalize(features, p=1, dim=1)
        elif self.__feat_norm == "col":
            features = self.col_normalization(features)
        features = features.to(device)

        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(0) # remove incompelte self-loops before adding self-loops
        adj_matrix_sl = adj_matrix + sp.eye(*adj_matrix.shape)
        adj_orig = sparse_mx_to_pyg_sparse_tensor(adj_matrix_sl).to_dense()
        adj_norm_matrix = LaplacianGraphOp()._construct_adj(adj_matrix)
        adj_norm = sparse_mx_to_pyg_sparse_tensor(adj_norm_matrix).to(device)
        adj = sparse_mx_to_pyg_sparse_tensor(adj_matrix).to(device)

        if self.__minibatch is False:
            adj_orig = adj_orig.to(device)

        return features, adj_orig, adj, adj_norm

    @staticmethod
    def sample_adj(adj_logits, temp):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=temp, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    @staticmethod
    def sample_adj_add_bernoulli(adj_logits, adj_orig, alpha, temp):
        edge_probs = adj_logits / (torch.max(adj_logits) + 1e-5)
        edge_probs = alpha * edge_probs + (1-alpha) * adj_orig
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=temp, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    @staticmethod
    def sample_adj_add_round(adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha * edge_probs + (1-alpha) * adj_orig
        # sampling
        adj_sampled = RoundNoGradient.apply(edge_probs)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    @staticmethod
    def sample_adj_random(adj_logits):
        adj_rand = torch.rand(adj_logits.size())
        adj_rand = adj_rand.triu(1)
        adj_rand = torch.round(adj_rand)
        adj_rand = adj_rand + adj_rand.T
        return adj_rand

    @staticmethod
    def sample_adj_edge(adj_logits, adj_orig, change_frac):
        adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
        n_edges = adj.nonzero().size(0)
        n_change = int(n_edges * change_frac / 2)
        # take only the upper triangle
        edge_probs = adj_logits.triu(1)
        edge_probs = edge_probs - torch.min(edge_probs)
        edge_probs = edge_probs / torch.max(edge_probs)
        adj_inverse = 1 - adj
        # get edges to be removed
        mask_rm = edge_probs * adj
        nz_mask_rm = mask_rm[mask_rm>0]
        if len(nz_mask_rm) > 0:
            n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
            thresh_rm = torch.topk(mask_rm[mask_rm>0], n_rm, largest=False)[0][-1]
            mask_rm[mask_rm > thresh_rm] = 0
            mask_rm = CeilNoGradient.apply(mask_rm)
            mask_rm = mask_rm + mask_rm.T
        # remove edges
        adj_new = adj - mask_rm
        # get edges to be added
        mask_add = edge_probs * adj_inverse
        nz_mask_add = mask_add[mask_add>0]
        if len(nz_mask_add) > 0:
            n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
            thresh_add = torch.topk(mask_add[mask_add>0], n_add, largest=True)[0][-1]
            mask_add[mask_add < thresh_add] = 0
            mask_add = CeilNoGradient.apply(mask_add)
            mask_add = mask_add + mask_add.T
        # add edges
        adj_new = adj_new + mask_add
        return adj_new

    def forward(self, adj_norm, adj_orig, features, nodes_batch=None):
        adj_logits = self.ep_net(adj_norm, features, nodes_batch)
        if self.__sample_type == "edge":
            adj_new = self.sample_adj_edge(adj_logits, adj_orig, self.__alpha)
        elif self.__sample_type == "add_round":
            adj_new = self.sample_adj_add_round(adj_logits, adj_orig, self.__alpha)
        elif self.__sample_type == "rand":
            adj_new = self.sample_adj_random(adj_logits)
        elif self.__sample_type == "add_sample":
            if self.__alpha == 1:
                adj_new = self.sample_adj(adj_logits, self.__temperature)
            else:
                adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.__alpha, self.__temperature)
        
        row, col = adj_new.nonzero(as_tuple=True)
        edge_index = torch.vstack([row, col])
        if nodes_batch is not None:
            nc_logits = self.nc_net(features[nodes_batch], edge_index)
        else:
            nc_logits = self.nc_net(features, edge_index)

        return nc_logits, adj_logits

class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, in_dim, hidden_dim, emb_size, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.activation = activation
        self.gcn_base = GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=False, bias=False)
        self.gcn_mean = GCNConv(hidden_dim, emb_size, add_self_loops=False, normalize=False, bias=False)
        self.gcn_logstd = GCNConv(hidden_dim, emb_size, add_self_loops=False, normalize=False, bias=False)

    def reset_parameters(self):
        self.gcn_base.reset_parameters()
        self.gcn_mean.reset_parameters()
        self.gcn_logstd.reset_parameters()

    def forward(self, adj, features, nodes_batch=None):
        # GCN encoder
        hidden = self.gcn_base(features, adj)
        self.mean = self.activation(self.gcn_mean(hidden, adj))
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = self.activation(self.gcn_logstd(hidden, adj))
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise * torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        if nodes_batch is not None:
            Z = Z[nodes_batch]
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


class GAugM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, gnn_type, rm_pct, add_pct, choose_idx, gae=False, dropout=0.5, activation=F.relu, feat_norm='none', **kwargs):
        super(GAugM, self).__init__()

        self.__feat_norm = feat_norm
        self.__rm_pct = rm_pct
        self.__add_pct = add_pct
        self.__choose_idx = choose_idx
        self.__gae = gae
        if isinstance(activation, str):
            activation = getattr(F, activation)
        gnn_backbone = {"gcn": GCN, "gsage": SAGE, "gat": GAT}
        if gnn_type == "gat":
            if kwargs.get("n_heads"):
                n_heads = list(map(lambda x: int(x), kwargs["n_heads"].split(",")))
            else:
                n_heads = [8] * (n_layers - 1) + [1]
            kwargs.update({"n_heads": n_heads})

        self.nc_net = gnn_backbone.get(gnn_type)(in_dim, hidden_dim, n_classes, n_layers=n_layers, dropout=dropout, activation=activation, **kwargs)

    def reset_parameters(self):
        self.nc_net.reset_parameters()
        
    @staticmethod
    def sample_graph_det(adj_orig, adj_pred, remove_pct, add_pct):
        if remove_pct == 0 and add_pct == 0:
            return copy.deepcopy(adj_orig)

        orig_upper = sp.triu(adj_orig, 1)
        n_edges = orig_upper.nnz
        edges = np.asarray(orig_upper.nonzero()).T

        if remove_pct:
            n_remove = int(n_edges * remove_pct / 100)
            pos_probs = adj_pred[edges.T[0], edges.T[1]]
            e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
            mask = np.ones(len(edges), dtype=bool)
            mask[e_index_2b_remove] = False
            edges_pred = edges[mask]
        else:
            edges_pred = edges

        if add_pct:
            n_add = int(n_edges * add_pct / 100)
            # deep copy to avoid modifying adj_pred
            adj_probs = np.array(adj_pred)
            # make the probabilities of the lower half to be zero (including diagonal)
            adj_probs[np.tril_indices(adj_probs.shape[0])] = 0
            # make the probabilities of existing edges to be zero
            adj_probs[edges.T[0], edges.T[1]] = 0
            all_probs = adj_probs.reshape(-1)
            e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
            new_edges = []
            for index in e_index_2b_add:
                i = int(index / adj_probs.shape[0])
                j = index % adj_probs.shape[0]
                new_edges.append([i, j])
            edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
        adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
        adj_pred = adj_pred + adj_pred.T

        return adj_pred

    def preprocess(self, adj_orig, features, adj_pred_dir, device):
        if self.__feat_norm == "row":
            features = F.normalize(features, p=1, dim=1)
        features = features.to(device)

        if self.__gae is True:
            adj_pred = pkl.load(open(os.path.join(adj_pred_dir, f"{self.__choose_idx}_logits_gae.pkl"), "rb"))
        else:
            adj_pred = pkl.load(open(os.path.join(adj_pred_dir, f"{self.__choose_idx}_logits.pkl"), "rb"))
        adj_pred = self.sample_graph_det(adj_orig, adj_pred, self.__rm_pct, self.__add_pct)
        adj_processed = sparse_mx_to_pyg_sparse_tensor(adj_pred).to(device)

        return adj_processed, features

    def forward(self, adj, features):
        return self.nc_net(features, adj)

