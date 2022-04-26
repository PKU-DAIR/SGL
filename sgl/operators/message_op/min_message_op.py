import torch

from sgl.operators.base_op import MessageOp


class MinMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MinMessageOp, self).__init__(start, end)
        self._aggr_type = "min"

    def _combine(self, feat_list):
        return torch.stack(feat_list[self._start:self._end], dim=0).min(dim=0)[0]
