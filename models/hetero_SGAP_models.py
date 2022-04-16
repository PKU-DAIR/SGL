from models.base_model import BaseHeteroSGAPModel, FastBaseHeteroSGAPModel
from models.graph_operator import LaplacianGraphOp
from models.message_operator import ProjectedConcatMessageOp, LearnableWeightedMessageOp
from models.simple_models import OneDimConvolution, MultiLayerPerceptron, OneDimConvolutionWeightSharedAcrossFeatures, FastOneDimConvolution


class NARS_SIGN(BaseHeteroSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers, random_subgraph_num,
                 subgraph_edge_type_num):
        super(NARS_SIGN, self).__init__(prop_steps, feat_dim,
                                        num_classes, random_subgraph_num, subgraph_edge_type_num)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = ProjectedConcatMessageOp(
            0, prop_steps + 1, feat_dim, hidden_dim, num_layers)

        self._aggregator = OneDimConvolution(
            random_subgraph_num, prop_steps + 1, feat_dim)
        self._base_model = MultiLayerPerceptron(
            hidden_dim * (prop_steps + 1), hidden_dim, num_layers, num_classes)


class NARS_SIGN_WeightSharedAcrossFeatures(BaseHeteroSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers, random_subgraph_num,
                 subgraph_edge_type_num):
        super(NARS_SIGN_WeightSharedAcrossFeatures, self).__init__(prop_steps, feat_dim,
                                                                   num_classes, random_subgraph_num, subgraph_edge_type_num)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = ProjectedConcatMessageOp(
            0, prop_steps + 1, feat_dim, hidden_dim, num_layers)

        self._aggregator = OneDimConvolutionWeightSharedAcrossFeatures(
            random_subgraph_num, prop_steps + 1)
        self._base_model = MultiLayerPerceptron(
            hidden_dim * (prop_steps + 1), hidden_dim, num_layers, num_classes)


class NARS_SGC_WithLearnableWeights(BaseHeteroSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers, random_subgraph_num,
                 subgraph_edge_type_num):
        super(NARS_SGC_WithLearnableWeights, self).__init__(prop_steps, feat_dim,
                                                            num_classes, random_subgraph_num, subgraph_edge_type_num)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LearnableWeightedMessageOp(
            0, prop_steps + 1, 'simple_allow_neg', prop_steps)

        self._aggregator = OneDimConvolutionWeightSharedAcrossFeatures(
            random_subgraph_num, prop_steps + 1)
        self._base_model = MultiLayerPerceptron(
            feat_dim, hidden_dim, num_layers, num_classes)


class Fast_NARS_SGC_WithLearnableWeights(FastBaseHeteroSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers, random_subgraph_num,
                 subgraph_edge_type_num):
        super(Fast_NARS_SGC_WithLearnableWeights, self).__init__(prop_steps, feat_dim,
                                                            num_classes, random_subgraph_num, subgraph_edge_type_num)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)

        self._aggregator = FastOneDimConvolution(
            random_subgraph_num, prop_steps + 1)
        self._base_model = MultiLayerPerceptron(
            feat_dim, hidden_dim, num_layers, num_classes)
