import torch

from sgl.operators.base_op import MessageOp


class ConcatMessageOp(MessageOp):
    def __init__(self, start, end):
        super(ConcatMessageOp, self).__init__(start, end)
        self._aggr_type = "concat"

    def _combine(self, feat_list):
        return torch.hstack(feat_list[self._start:self._end])
