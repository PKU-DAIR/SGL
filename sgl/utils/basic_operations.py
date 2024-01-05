import torch
import numpy as np
from torch_sparse import SparseTensor

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_pyg_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a PyG SparseTensor"""
    sparse_mx = sparse_mx.tocoo()
    row = torch.from_numpy(sparse_mx.row).to(torch.long)
    col = torch.from_numpy(sparse_mx.col).to(torch.long)
    value = torch.from_numpy(sparse_mx.data)
    sparse_sizes = torch.Size(sparse_mx.shape)
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=sparse_sizes, is_sorted=True, trust_data=True)