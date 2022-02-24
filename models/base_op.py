import torch
import torch.nn as nn
from torch import Tensor


class GraphOp:
    def __init__(self, prop_steps):
        self._prop_steps = prop_steps
        self._adj = None

    def _construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self._adj = self._construct_adj(adj)

        # use torch.sparse temporarily
        if not (isinstance(self._adj, Tensor) and self._adj.is_sparse):
            raise TypeError("The adjacency matrix must be a sparse tensor!")
        elif not isinstance(feature, Tensor):
            raise TypeError("The feature matrix must be a tensor!")
        elif self._adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            prop_feat_list.append(torch.spmm(self._adj, prop_feat_list[-1]))
        return prop_feat_list


# Might include training parameters
class MessageOp(nn.Module):
    def __init__(self):
        super(MessageOp, self).__init__()
        self._aggr_type = None

    @property
    def aggr_type(self):
        return self._aggr_type

    def _combine(self, feat_list, *args):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self._combine(feat_list)
