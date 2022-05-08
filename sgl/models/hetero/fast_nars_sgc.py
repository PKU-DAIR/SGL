from sgl.models.base_model import FastBaseHeteroSGAPModel
from sgl.models.simple_models import MultiLayerPerceptron, FastOneDimConvolution
from sgl.operators.graph_op import LaplacianGraphOp


class Fast_NARS_SGC_WithLearnableWeights(FastBaseHeteroSGAPModel):
    def __init__(self, prop_steps, feat_dim, output_dim, hidden_dim, num_layers, random_subgraph_num):
        super(Fast_NARS_SGC_WithLearnableWeights, self).__init__(prop_steps, feat_dim, output_dim)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)

        self._aggregator = FastOneDimConvolution(
            random_subgraph_num, prop_steps + 1)
        self._base_model = MultiLayerPerceptron(
            feat_dim, hidden_dim, num_layers, output_dim)

    @property
    def subgraph_weight(self):
        return self._aggregator.subgraph_weight
