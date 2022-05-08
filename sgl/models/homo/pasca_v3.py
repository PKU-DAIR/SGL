from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import ResMultiLayerPerceptron
from sgl.operators.graph_op import LaplacianGraphOp, PprGraphOp
from sgl.operators.message_op import LearnableWeightedMessageOp, LastMessageOp
        

class PASCA_V3(BaseSGAPModel):
    def __init__(self, prop_steps, post_steps, feat_dim, output_dim, hidden_dim, num_layers):
        super(PASCA_V3, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LearnableWeightedMessageOp(1, prop_steps + 1, "gate", feat_dim)
        self._base_model = ResMultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim, 0.8)
        self._post_graph_op = PprGraphOp(post_steps, r=0.5, alpha=0.3)
        self._post_msg_op = LastMessageOp()
