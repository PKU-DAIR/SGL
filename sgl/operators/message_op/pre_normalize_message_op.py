from sgl.operators.base_op import PreMessageOp

import torch.nn.functional as F

class PreNormMessageOp(PreMessageOp):
    def __init__(self, p=1, dim=1):
        super(PreNormMessageOp, self).__init__(dim)
        self._p = p

    def _transform_x(self, x):
        return F.normalize(x, p=self._p, dim=self._dim)