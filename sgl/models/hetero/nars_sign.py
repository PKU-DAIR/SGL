from sgl.models.base_model import BaseHeteroSGAPModel
from sgl.models.simple_models import OneDimConvolution, MultiLayerPerceptron
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.operators.message_op import ProjectedConcatMessageOp


class NARS_SIGN(BaseHeteroSGAPModel):
    def __init__(self, prop_steps, feat_dim, output_dim, hidden_dim, num_layers, random_subgraph_num):
        super(NARS_SIGN, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = ProjectedConcatMessageOp(
            0, prop_steps + 1, feat_dim, hidden_dim, num_layers)

        self._aggregator = OneDimConvolution(
            random_subgraph_num, prop_steps + 1, feat_dim)
        self._base_model = MultiLayerPerceptron(
            hidden_dim * (prop_steps + 1), hidden_dim, num_layers, output_dim)
