import torch
from torch import Tensor
import numpy as np
import scipy.sparse as sp


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


def one_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError("This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[0]:
        raise ValueError("The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 1:
        raise ValueError("The weight list should be a 1d tensor!")

    feat_shape = feat_list[0].shape
    feat_reshape = torch.vstack([feat.view(1, -1).squeeze(0) for feat in feat_list])
    weighted_feat = (feat_reshape * weight_list).sum(dim=0).view(feat_shape)
    return weighted_feat


def two_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError("This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[1]:
        raise ValueError("The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 2:
        raise ValueError("The weight list should be a 2d tensor!")

    feat_reshape = torch.stack(feat_list, dim=2)
    weight_reshape = weight_list.unsqueeze(dim=2)
    weighted_feat = torch.bmm(feat_reshape, weight_reshape).squeeze(dim=2)
    return weighted_feat
