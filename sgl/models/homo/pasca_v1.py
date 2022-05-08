from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import ResMultiLayerPerceptron
from sgl.operators.graph_op import PprGraphOp
from sgl.operators.message_op import LearnableWeightedMessageOp
        

class PASCA_V1(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, output_dim, hidden_dim, num_layers):
        super(PASCA_V1, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = PprGraphOp(prop_steps, r=0.5, alpha=0.1)
        self._pre_msg_op = LearnableWeightedMessageOp(1, prop_steps + 1, "simple", feat_dim)
        self._base_model = ResMultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim, 0.8)
