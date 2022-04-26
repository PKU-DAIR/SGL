from models.base_model import BaseSGAPModel
from models.simple_models import MultiLayerPerceptron
from operators.graph_op import LaplacianGraphOp
from operators.message_op import ConcatMessageOp, ProjectedConcatMessageOp


# slightly different from the original design
class SIGN(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers):
        super(SIGN, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        # self._pre_msg_op = ProjectedConcatMessageOp(0, prop_steps + 1, feat_dim, hidden_dim)
        self._pre_msg_op = ConcatMessageOp(0, prop_steps + 1)
        self._base_model = MultiLayerPerceptron((prop_steps + 1) * feat_dim, hidden_dim, num_layers, num_classes)
