from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import MultiLayerPerceptron
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.operators.message_op import LearnableWeightedMessageOp


class GAMLP(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, output_dim, hidden_dim, num_layers):
        super(GAMLP, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LearnableWeightedMessageOp(0, prop_steps + 1, "jk", prop_steps, feat_dim)
        self._base_model = MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim)
