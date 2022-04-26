import torch
import torch.nn.functional as F
from torch.nn import ModuleList

from sgl.models.simple_models import MultiLayerPerceptron
from sgl.operators.base_op import MessageOp


class ProjectedConcatMessageOp(MessageOp):
    def __init__(self, start, end, feat_dim, hidden_dim, num_layers):
        super(ProjectedConcatMessageOp, self).__init__(start, end)
        self._aggr_type = "proj_concat"

        self.__learnable_weight = ModuleList()
        for _ in range(end - start):
            self.__learnable_weight.append(MultiLayerPerceptron(
                feat_dim, hidden_dim, num_layers, hidden_dim))

    def _combine(self, feat_list):
        adopted_feat_list = feat_list[self._start:self._end]

        concat_feat = self.__learnable_weight[0](adopted_feat_list[0])
        for i in range(1, self._end - self._start):
            transformed_feat = F.relu(
                self.__learnable_weight[i](adopted_feat_list[i]))
            concat_feat = torch.hstack((concat_feat, transformed_feat))

        return concat_feat
