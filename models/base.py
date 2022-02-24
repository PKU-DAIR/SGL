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


class BaseSGAPModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, num_classes):
        super(BaseSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._num_classes = num_classes

        self._pre_graph_op, self._pre_msg_op = None, None
        self._post_graph_op, self._post_msg_op = None, None
        self._base_model = None

        self._processed_feat_list = None

    def preprocess(self, adj, feature):
        self._processed_feat_list = self._pre_graph_op.propagate(adj, feature)

    def postprocess(self, output):
        if self._post_graph_op is not None:
            if self._post_msg_op.aggr_type == "learnable_weighted":
                raise ValueError("Learnable weighted message operator is not supported in the post-processing phase!")
            output = self._post_msg_op(self._post_graph_op(output))

        return output

    # a wrapper of the forward function
    def train_model(self, adj, feature):
        return self.forward(adj, feature)

    def forward(self, adj, feature):
        processed_feature = self._pre_msg_op.aggregate(self._processed_feat_list)
        output = self._base_model(processed_feature)

        return output
