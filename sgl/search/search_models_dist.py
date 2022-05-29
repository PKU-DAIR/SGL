from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import LogisticRegression, MultiLayerPerceptron, ResMultiLayerPerceptron
from sgl.operators.graph_op import LaplacianGraphOp, PprGraphOp
from sgl.operators.message_op import LastMessageOp, ConcatMessageOp, MeanMessageOp, SimpleWeightedMessageOp, \
    LearnableWeightedMessageOp, IterateLearnableWeightedMessageOp, SumMessageOp, MaxMessageOp, MinMessageOp


class SearchModelDist(BaseSGAPModel):
    def __init__(self, arch, feat_dim, output_dim, hidden_dim):
        prop_steps = arch[0]
        prop_types = arch[1]
        mesg_types = arch[2]
        num_layers = arch[3]
        post_steps = arch[4]
        post_types = arch[5]
        pmsg_types = arch[6]
        super(SearchModelDist, self).__init__(prop_steps, feat_dim, output_dim)
        
        if prop_types == 1:
            self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        elif prop_types == 2:
            self._pre_graph_op = PprGraphOp(prop_steps, r=0.5, alpha=0.1)
        elif prop_types == 3:
            self._pre_graph_op = PprGraphOp(prop_steps, r=0.5, alpha=0.2)
        elif prop_types == 4:
            self._pre_graph_op = PprGraphOp(prop_steps, r=0.5, alpha=0.3)        

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
            self._pre_msg_op = LearnableWeightedMessageOp(0, prop_steps, "gate", feat_dim)
        elif mesg_types == 8:
            self._pre_msg_op = IterateLearnableWeightedMessageOp(0, prop_steps + 1, "recursive", feat_dim)
        

        if num_layers == 1:
            self._base_model = LogisticRegression(feat_dim, output_dim)
        elif num_layers == 2:
            self._base_model = MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim)
        else:
            self._base_model = ResMultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim)

        if post_types != 0 and post_steps != 0:
            if post_types == 1:
                self._post_graph_op = LaplacianGraphOp(post_steps, r=0.5)
            elif post_types == 2:
                self._post_graph_op = PprGraphOp(post_steps, r=0.5, alpha=0.1)
            elif post_types == 3:
                self._post_graph_op = PprGraphOp(post_steps, r=0.5, alpha=0.2)
            elif post_types == 4:
                self._post_graph_op = PprGraphOp(post_steps, r=0.5, alpha=0.3)
            
            if pmsg_types == 0:
                self._post_msg_op = LastMessageOp()
            elif pmsg_types == 1:
                self._post_msg_op = MeanMessageOp(start=0, end=post_steps + 1)
            elif pmsg_types == 2:
                self._post_msg_op = SumMessageOp(start=0, end=post_steps + 1)
            elif pmsg_types == 3:
                self._post_msg_op = MaxMessageOp(start=0, end=post_steps + 1)
            elif pmsg_types == 4:
                self._post_msg_op = MinMessageOp(start=0, end=post_steps + 1)
            elif pmsg_types == 5:
                self._post_msg_op = SimpleWeightedMessageOp(0, post_steps + 1, "alpha", 0.85)

    def preprocess(self, adj, feature):
        if self._pre_graph_op is not None:
            self._processed_feat_list = self._pre_graph_op.propagate(
                adj, feature)
        else:
            self._processed_feat_list = [feature]

    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.evaluate_forward(idx, device)

    def evaluate_forward(self, idx, device):   
        transferred_feat_list = [feat[idx].to(
                device) for feat in self._processed_feat_list]
        processed_feature = self._pre_msg_op.aggregate(
            transferred_feat_list)

        output = self._base_model(processed_feature)
        return output
            
    def forward(self, transferred_feat_list):   
        processed_feature = self._pre_msg_op.aggregate(
            transferred_feat_list)
            
        output = self._base_model(processed_feature)
        return output
