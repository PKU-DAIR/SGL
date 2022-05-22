import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
import math

def LogeCrossEntropy(pred, target, epsilon= 1.0 - math.log(2)):
    loss = F.cross_entropy(pred, target)
    loss = torch.log(epsilon + loss) - math.log(epsilon)
    return torch.mean(loss)

def LogeBinaryCrossEntropyWithLogits(pred, target, epsilon= 1.0 - math.log(2)):
    loss = F.binary_cross_entropy_with_logits(pred, target)
    loss = torch.log(epsilon + loss) - math.log(epsilon)
    return loss

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
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

@torch.no_grad()
def label_propagation(labels, adj, num_layers, alpha, post_process = lambda x: x.clamp_(0., 1.), mask=None):
    if labels.dtype == torch.long:
        labels = F.one_hot(labels.reshape(-1)).to(torch.float)

    out = labels.clone()
    if mask is not None:
        out = torch.zeros_like(labels)
        out[mask] = labels[mask]

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
    
    # record H_0
    res = (1 - alpha) * out
    for _ in range(num_layers):
        out = alpha * torch.spmm(adj_tensor, out) + res
        out = post_process(out)
    
    return out
