from models.base_model import BaseSGAPModel

from models.graph_operator import LaplacianGraphOp
from models.message_operator import LastMessageOp, ConcatMessageOp, MeanMessageOp, SimpleWeightedMessageOp, \
    LearnableWeightedMessageOp
from models.simple_models import LogisticRegression, MultiLayerPerceptron


class SGC(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes):
        super(SGC, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LastMessageOp()
        self._base_model = LogisticRegression(feat_dim, num_classes)


# slightly different from the original design
class SIGN(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers):
        super(SIGN, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = ConcatMessageOp()
        self._base_model = MultiLayerPerceptron((prop_steps + 1) * feat_dim, hidden_dim, num_layers, num_classes)


class SSGC(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes):
        super(SSGC, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = MeanMessageOp()
        self._base_model = LogisticRegression(feat_dim, num_classes)


class GBP(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers, r=0.5, alpha=0.85):
        super(GBP, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = SimpleWeightedMessageOp("alpha", alpha)
        self._base_model = MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, num_classes)


class GAMLP(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes, hidden_dim, num_layers):
        super(GAMLP, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LearnableWeightedMessageOp("jk", prop_steps, feat_dim)
        self._base_model = MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, num_classes)
