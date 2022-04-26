from sgl.operators.base_op import MessageOp


class SumMessageOp(MessageOp):
    def __init__(self, start, end):
        super(SumMessageOp, self).__init__(start, end)
        self._aggr_type = "sum"

    def _combine(self, feat_list):
        return sum(feat_list[self._start:self._end])
