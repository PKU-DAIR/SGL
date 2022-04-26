import torch
import torch.nn.functional as F
from torch.nn import Linear

from sgl.operators.base_op import MessageOp


class IterateLearnableWeightedMessageOp(MessageOp):

    # 'recursive' needs one additional parameter 'feat_dim'
    def __init__(self, start, end, combination_type, *args):
        super(IterateLearnableWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = "iterate_learnable_weighted"

        if combination_type not in ["recursive"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'recursive'.")
        self.__combination_type = combination_type

        self.__learnable_weight = None
        if combination_type == "recursive":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the recursive iterate weighted aggregator!")
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim + feat_dim, 1)

    def _combine(self, feat_list):
        weight_list = None
        if self.__combination_type == "recursive":
            weighted_feat = feat_list[self._start]
            for i in range(self._start, self._end):
                weights = torch.sigmoid(self.__learnable_weight(
                    torch.hstack((feat_list[i], weighted_feat))))
                if i == self._start:
                    weight_list = weights
                else:
                    weight_list = torch.hstack((weight_list, weights))
                weight_list = F.softmax(weight_list, dim=1)

                weighted_feat = torch.mul(
                    feat_list[self._start], weight_list[:, 0].view(-1, 1))
                for j in range(1, i + 1):
                    weighted_feat = weighted_feat + \
                                    torch.mul(feat_list[self._start + j],
                                              weight_list[:, j].view(-1, 1))

        else:
            raise NotImplementedError

        return weighted_feat
