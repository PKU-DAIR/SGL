from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import LogisticRegression, MultiLayerPerceptron
from sgl.operators.graph_op import LaplacianGraphOp, PprGraphOp
from sgl.operators.message_op import LastMessageOp, ConcatMessageOp, MeanMessageOp, SimpleWeightedMessageOp, \
    LearnableWeightedMessageOp, IterateLearnableWeightedMessageOp, SumMessageOp, MaxMessageOp, MinMessageOp


class SearchModel(BaseSGAPModel):
    def __init__(self, arch, feat_dim, num_classes, hidden_dim):
        prop_steps = arch[0]
        prop_types = arch[1]
        mesg_types = arch[2]
        num_layers = arch[3]
        post_steps = arch[4]
        post_types = arch[5]
        pmsg_types = arch[6]
        super(SearchModel, self).__init__(prop_steps, feat_dim, num_classes)

        if prop_types == 0:
            self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        else:
            self._pre_graph_op = PprGraphOp(prop_steps, r=0.5, alpha=0.15)

        if mesg_types == 0:
            self._pre_msg_op = LastMessageOp()
        elif mesg_types == 1:
            self._pre_msg_op = ConcatMessageOp(start=0, end=prop_steps + 1)
            feat_dim *= prop_steps + 1
        elif mesg_types == 2:
            self._pre_msg_op = MeanMessageOp(start=0, end=prop_steps + 1)
        elif mesg_types == 3:
            self._pre_msg_op = SumMessageOp(start=0, end=prop_steps + 1)
        elif mesg_types == 4:
            self._pre_msg_op = MaxMessageOp(start=0, end=prop_steps + 1)
        elif mesg_types == 5:
            self._pre_msg_op = MinMessageOp(start=0, end=prop_steps + 1)
        elif mesg_types == 6:
            self._pre_msg_op = SimpleWeightedMessageOp(0, prop_steps + 1, "alpha", 0.85)
        elif mesg_types == 7:
            self._pre_msg_op = LearnableWeightedMessageOp(0, prop_steps + 1, "gate", feat_dim)
        elif mesg_types == 8:
            self._pre_msg_op = IterateLearnableWeightedMessageOp(0, prop_steps + 1, "recursive", feat_dim)

        if num_layers == 1:
            self._base_model = LogisticRegression(feat_dim, num_classes)
        else:
            self._base_model = MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, num_classes)

        if post_types != 0 and post_steps != 0:
            if post_types == 1:
                self._post_graph_op = LaplacianGraphOp(post_steps, r=0.5)
            else:
                self._post_graph_op = PprGraphOp(post_steps, r=0.5, alpha=0.15)

            if pmsg_types == 0:
                self._post_msg_op = LastMessageOp()
            elif pmsg_types == 1:
                self._post_msg_op = MeanMessageOp(start=0, end=post_steps + 1)
            elif mesg_types == 2:
                self._post_msg_op = SumMessageOp(start=0, end=post_steps + 1)
            elif mesg_types == 3:
                self._post_msg_op = MaxMessageOp(start=0, end=post_steps + 1)
            elif mesg_types == 4:
                self._post_msg_op = MinMessageOp(start=0, end=post_steps + 1)
            elif mesg_types == 5:
                self._post_msg_op = SimpleWeightedMessageOp(0, post_steps + 1, "alpha", 0.85)
