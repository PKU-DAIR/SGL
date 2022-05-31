import numpy as np
import platform
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import Tensor

from sgl.operators.utils import csr_sparse_dense_matmul, cuda_csr_sparse_dense_matmul


class GraphOp:
    def __init__(self, prop_steps):
        self._prop_steps = prop_steps
        self._adj = None

    def _construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self._adj = self._construct_adj(adj)

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self._adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            if platform.system() == "Linux":
                feat_temp = csr_sparse_dense_matmul(self._adj, prop_feat_list[-1])
            else:
                feat_temp = self._adj.dot(prop_feat_list[-1])
            prop_feat_list.append(feat_temp)
        return [torch.FloatTensor(feat) for feat in prop_feat_list]


# Might include training parameters
class MessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(MessageOp, self).__init__()
        self._aggr_type = None
        self._start, self._end = start, end

    @property
    def aggr_type(self):
        return self._aggr_type

    def _combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self._combine(feat_list)
