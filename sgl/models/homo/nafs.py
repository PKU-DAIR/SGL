from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import LogisticRegression
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.operators.message_op import OverSmoothDistanceWeightedOp


class NAFS(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, r=0.5):
        super(NAFS, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=r)
        self._pre_msg_op = OverSmoothDistanceWeightedOp()
        self._base_model = LogisticRegression(feat_dim, num_classes)