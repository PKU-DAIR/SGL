import torch
from torch import Tensor
from torch.nn import Parameter

from base import MessageOp
from utils import one_dim_weighted_add, two_dim_weighted_add


class LastMessageOp(MessageOp):
    def __init__(self):
        super(LastMessageOp, self).__init__()
        self._aggr_type = "last"

    def _combine(self, feat_list, *args):
        return feat_list[-1]


class SumMessageOp(MessageOp):
    def __init__(self):
        super(SumMessageOp, self).__init__()
        self._aggr_type = "sum"

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the sum aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        return sum(feat_list[start:end])


class MeanMessageOp(MessageOp):
    def __init__(self):
        super(MeanMessageOp, self).__init__()
        self._aggr_type = "mean"

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the mean aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        return sum(feat_list[start:end]) / (end - start)


class MaxMessageOp(MessageOp):
    def __init__(self):
        super(MaxMessageOp, self).__init__()
        self._aggr_type = "max"

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the maximum aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        return torch.stack(feat_list[start:end], dim=0).max(dim=0)[0]


class MinMessageOp(MessageOp):
    def __init__(self):
        super(MinMessageOp, self).__init__()
        self._aggr_type = "min"

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the minimum aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        return torch.stack(feat_list[start:end], dim=0).min(dim=0)[0]


class ConcatMessageOp(MessageOp):
    def __init__(self):
        super(ConcatMessageOp, self).__init__()
        self._aggr_type = "concat"

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the concat aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        return torch.hstack(feat_list[start:end])


class SimpleWeightedMessageOp(MessageOp):

    # 'alpha' needs one additional parameter 'alpha';
    # 'hand_crafted' needs one additional parameter 'weight_list'
    def __init__(self, combination_type, *args):
        super(SimpleWeightedMessageOp, self).__init__()
        self._aggr_type = "simple_weighted"

        if combination_type not in ["alpha", "hand_crafted"]:
            raise ValueError("Invalid weighted combination type! Type must be 'alpha' or 'hand_crafted'.")
        self.__combination_type = combination_type

        if len(args) != 1:
            raise ValueError("Invalid parameter numbers for the simple weighted aggregator!")
        self.__alpha, self.__weight_list = None, None
        if combination_type is "alpha":
            self.__alpha = args[0]
            if not isinstance(self.__alpha, float):
                raise TypeError("The alpha must be a float!")
            elif self.__alpha > 1 or self.__alpha < 0:
                raise ValueError("The alpha must be a float in [0,1]!")

        elif combination_type is "hand_crafted":
            self.__weight_list = args[0]
            if isinstance(self.__weight_list, list):
                weight_list = torch.FloatTensor(self.__weight_list)
            elif not isinstance(self.__weight_list, (list, Tensor)):
                raise TypeError("The input weight list must be a list or a tensor!")

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the simple weighted aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        if self.__combination_type == "alpha":
            self.__weight_list = [self.__alpha]
            for _ in range(len(feat_list) - 1):
                self.__weight_list.append((1 - self.__alpha) * self.__weight_list[-1])
            self.__weight_list = torch.FloatTensor(self.__weight_list[start:end])

        elif self.__combination_type == "hand_crafted":
            pass
        else:
            raise NotImplementedError

        weighted_feat = one_dim_weighted_add(feat_list[start:end], weight_list=self.__weight_list)
        return weighted_feat


class LearnableWeightedMessageOp(MessageOp):

    # 'simple' needs one additional parameter 'prop_steps';
    # 'gate' needs one additional parameter 'feat_dim';
    # 'ori_ref' needs one additional parameter 'feat_dim';
    # 'jk' needs two additional parameter 'prop_steps' and 'feat_dim'
    def __init__(self, combination_type, *args):
        super(LearnableWeightedMessageOp, self).__init__()
        self._aggr_type = "learnable_weighted"

        if combination_type not in ["simple", "gate", "ori_ref", "jk"]:
            raise ValueError("Invalid weighted combination type! Type must be 'simple', 'gate', 'ori_ref' or 'jk'.")
        self.__combination_type = combination_type

        self.__learnable_weight = None
        if combination_type == "simple":
            if len(args) != 1:
                raise ValueError("Invalid parameter numbers for the simple learnable weighted aggregator!")
            prop_steps = args[0]
            self.__learnable_weight = Parameter(torch.FloatTensor(prop_steps + 1))

        elif combination_type == "gate":
            if len(args) != 1:
                raise ValueError("Invalid parameter numbers for the gate learnable weighted aggregator!")
            feat_dim = args[0]
            self.__learnable_weight = Parameter(torch.FloatTensor(feat_dim))

        elif combination_type == "ori_ref":
            if len(args) != 1:
                raise ValueError("Invalid parameter numbers for the ori_ref learnable weighted aggregator!")
            feat_dim = args[0]
            self.__learnable_weight = Parameter(torch.FloatTensor(feat_dim + feat_dim))

        elif combination_type == "jk":
            if len(args) != 2:
                raise ValueError("Invalid parameter numbers for the jk learnable weighted aggregator!")
            prop_steps, feat_dim = args[0], args[1]
            self.__learnable_weight = Parameter(torch.FloatTensor(feat_dim + (prop_steps + 1) * feat_dim))

    # two additional parameters, start and end, might be used
    def _combine(self, feat_list, *args):
        start, end = None, None
        if len(args) == 0:
            start, end = 0, len(feat_list)
        elif len(args) == 2:
            start, end = args[0], args[1] + 1
        elif len(args) not in [0, 2]:
            raise ValueError("Invalid parameter numbers for the learnable weighted aggregator!")
        elif args[0] < 0 or args[1] >= len(feat_list):
            raise ValueError("Invalid value for 'start' or 'end'!")

        weight_list = None
        if self.__combination_type == "simple":
            weight_list = self.__learnable_weight[start:end]

        elif self.__combination_type == "gate":
            adopted_feat_list = torch.vstack(feat_list[start:end])
            weight_list = torch.mm(adopted_feat_list, self.__learnable_weight).view(-1, end - start)

        elif self.__combination_type == "ori_ref":
            reference_feat = feat_list[0].repeat(end - start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(feat_list[start:end])))
            weight_list = torch.mm(adopted_feat_list, self.__learnable_weight).view(-1, end - start)

        elif self.__combination_type == "jk":
            reference_feat = torch.hstack(feat_list).repeat(end - start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(feat_list[start:end])))
            weight_list = torch.mm(adopted_feat_list, self.__learnable_weight).view(-1, end - start)
        else:
            raise NotImplementedError

        weighted_feat = None
        if self.__combination_type == "simple":
            weighted_feat = one_dim_weighted_add(feat_list[start:end], weight_list=weight_list)
        elif self.__combination_type in ["gate", "ori_ref", "jk"]:
            weighted_feat = two_dim_weighted_add(feat_list[start:end], weight_list=weight_list)
        else:
            raise NotImplementedError

        return weighted_feat
