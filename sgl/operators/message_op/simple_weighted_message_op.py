import torch
from torch import Tensor

from sgl.operators.base_op import MessageOp
from sgl.operators.utils import one_dim_weighted_add


class SimpleWeightedMessageOp(MessageOp):

    # 'alpha' needs one additional parameter 'alpha';
    # 'hand_crafted' needs one additional parameter 'weight_list'
    def __init__(self, start, end, combination_type, *args):
        super(SimpleWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = "simple_weighted"

        if combination_type not in ["alpha", "hand_crafted"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'alpha' or 'hand_crafted'.")
        self.__combination_type = combination_type

        if len(args) != 1:
            raise ValueError(
                "Invalid parameter numbers for the simple weighted aggregator!")
        self.__alpha, self.__weight_list = None, None
        if combination_type == "alpha":
            self.__alpha = args[0]
            if not isinstance(self.__alpha, float):
                raise TypeError("The alpha must be a float!")
            elif self.__alpha > 1 or self.__alpha < 0:
                raise ValueError("The alpha must be a float in [0,1]!")

        elif combination_type == "hand_crafted":
            self.__weight_list = args[0]
            if isinstance(self.__weight_list, list):
                self.__weight_list = torch.FloatTensor(self.__weight_list)
            elif not isinstance(self.__weight_list, (list, Tensor)):
                raise TypeError(
                    "The input weight list must be a list or a tensor!")

    def _combine(self, feat_list):
        if self.__combination_type == "alpha":
            self.__weight_list = [self.__alpha]
            for _ in range(len(feat_list) - 1):
                self.__weight_list.append(
                    (1 - self.__alpha) * self.__weight_list[-1])
            self.__weight_list = torch.FloatTensor(
                self.__weight_list[self._start:self._end])

        elif self.__combination_type == "hand_crafted":
            pass
        else:
            raise NotImplementedError

        weighted_feat = one_dim_weighted_add(
            feat_list[self._start:self._end], weight_list=self.__weight_list)
        return weighted_feat
