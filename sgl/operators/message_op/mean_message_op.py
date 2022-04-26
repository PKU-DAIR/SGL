from sgl.operators.base_op import MessageOp


class MeanMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MeanMessageOp, self).__init__(start, end)
        self._aggr_type = "mean"

    def _combine(self, feat_list):
        return sum(feat_list[self._start:self._end]) / (self._end - self._start)
