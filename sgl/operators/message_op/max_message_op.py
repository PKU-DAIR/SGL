import torch

from sgl.operators.base_op import MessageOp


class MaxMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MaxMessageOp, self).__init__(start, end)
        self._aggr_type = "max"

    def _combine(self, feat_list):
        return torch.stack(feat_list[self._start:self._end], dim=0).max(dim=0)[0]
