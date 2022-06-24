from sgl.models.base_model import BaseSGAPModelDist
from sgl.models.simple_models import LogisticRegression
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.operators.message_op import LastMessageOp


class SGCDist(BaseSGAPModelDist):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(SGCDist, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LastMessageOp()
        self._base_model = LogisticRegression(feat_dim, output_dim)
