from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import IdenticalMapping
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.operators.message_op import OverSmoothDistanceWeightedOp


class NAFS(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(NAFS, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = OverSmoothDistanceWeightedOp()
        self._base_model = IdenticalMapping()
