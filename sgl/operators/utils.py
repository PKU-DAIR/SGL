import os.path as osp
import numpy as np
import numpy.ctypeslib as ctl
import scipy.sparse as sp
import torch
from ctypes import c_int
from torch import Tensor


def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)


def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("./csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)


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
    weighted_feat = (feat_reshape * weight_list.view(-1, 1)).sum(dim=0).view(feat_shape)
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
