import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from sgl.models.simple_models import GCNConv, GCN, SAGE, GAT
from sgl.models.homo.gda.utils import RoundNoGradient, CeilNoGradient
from sgl.utils import sparse_mx_to_torch_sparse_tensor

class GAug(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 emb_size,
                 n_classes,
                 n_layers,
                 dropout,
                 gnnlayer_type,
                 activation=F.relu,
                 temperature=1,
                 gae=False,
                 alpha=1,
                 feat_norm="row",
                 sample_type="add_sample",
                 **kwargs):
        super(GAug, self).__init__()
        self.__temperature = temperature
        self.__gnnlayer_type = gnnlayer_type
        self.__alpha = alpha
        self.__sample_type = sample_type
        # edge prediction network
        self.__gae = gae
        self.__feat_norm = feat_norm
        self.ep_net = VGAE(in_dim, hidden_dim, emb_size, activation, gae=gae)
        # node classification network      
        select_model = {"gcn": GCN, "gsage": SAGE, "gat": GAT}
        if gnnlayer_type == 'gat':
            if kwargs.get("n_heads"):
                n_heads = list(map(lambda x: int(x), kwargs["n_heads"].split(",")))
            else:
                n_heads = [8] * (n_layers - 1) + [1]
            kwargs.update({"n_heads": n_heads})
            activation = F.elu

        self.nc_net = select_model.get(gnnlayer_type)(in_dim, hidden_dim, n_classes, nlayers=n_layers, dropout=dropout, activation=activation, **kwargs)
    
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
    
    def preprocess(self, features, adj_matrix, device):
        if self.__feat_norm == 'row':
            features = F.normalize(features, p=1, dim=1)
        elif self.__feat_norm == 'col':
            features = self.col_normalization(features)
        features = features.to(device)

        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        adj_orig = sparse_mx_to_torch_sparse_tensor(adj_matrix).to_dense().to(device)
        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm_matrix = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm_matrix)
        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        if self.__gnnlayer_type == 'gcn':
            adj = sparse_mx_to_torch_sparse_tensor(adj_norm_matrix)
        elif self.__gnnlayer_type == 'gsage':
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix)
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix_noselfloop / adj_matrix_noselfloop.sum(1))
            adj = sparse_mx_to_torch_sparse_tensor(adj_matrix_noselfloop)
        elif self.__gnnlayer_type == 'gat':
            adj = torch.FloatTensor(adj_matrix.todense())
        
        adj_norm = adj_norm.to(device)
        adj = adj.to(device)

        return features, adj_orig, adj_norm, adj
    
    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.__temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha * edge_probs + (1-alpha) * adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.__temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_round(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha * edge_probs + (1-alpha) * adj_orig
        # sampling
        adj_sampled = RoundNoGradient.apply(edge_probs)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_random(self, adj_logits):
        adj_rand = torch.rand(adj_logits.size())
        adj_rand = adj_rand.triu(1)
        adj_rand = torch.round(adj_rand)
        adj_rand = adj_rand + adj_rand.T
        return adj_rand

    def sample_adj_edge(self, adj_logits, adj_orig, change_frac):
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

    def normalize_adj(self, adj):
        if self.__gnnlayer_type == 'gcn':
            adj.fill_diagonal_(1)
            # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(adj.device)
            adj = D_norm @ adj @ D_norm
        elif self.__gnnlayer_type == 'gat':
            adj.fill_diagonal_(1)
        elif self.__gnnlayer_type == 'gsage':
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj

    def forward(self, adj, adj_orig, features):
        adj_logits = self.ep_net(adj, features)
        if self.__sample_type == 'edge':
            adj_new = self.sample_adj_edge(adj_logits, adj_orig, self.__alpha)
        elif self.__sample_type == 'add_round':
            adj_new = self.sample_adj_add_round(adj_logits, adj_orig, self.__alpha)
        elif self.__sample_type == 'rand':
            adj_new = self.sample_adj_random(adj_logits)
        elif self.__sample_type == 'add_sample':
            if self.__alpha == 1:
                adj_new = self.sample_adj(adj_logits)
            else:
                adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.__alpha)
        adj_new_normed = self.normalize_adj(adj_new)
        nc_logits = self.nc_net(features, adj_new_normed)
        return nc_logits, adj_logits


class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, in_dim, hidden_dim, emb_size, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.activation = activation
        self.gcn_base = GCNConv(in_dim, hidden_dim, bias=False)
        self.gcn_mean = GCNConv(hidden_dim, emb_size, bias=False)
        self.gcn_logstd = GCNConv(hidden_dim, emb_size, bias=False)

    def forward(self, adj, features):
        # GCN encoder
        hidden = self.gcn_base(features, adj, )
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
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits