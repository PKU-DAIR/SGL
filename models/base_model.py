import torch.nn as nn
import torch.nn.functional as F


class BaseSGAPModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, num_classes):
        super(BaseSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._num_classes = num_classes

        self._pre_graph_op, self._pre_msg_op = None, None
        self._post_graph_op, self._post_msg_op = None, None
        self._base_model = None

        self._processed_feat_list = None
        self._processed_feature = None
        self._pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self._pre_graph_op is not None:
            self._processed_feat_list = self._pre_graph_op.propagate(adj, feature)
            if self._pre_msg_op.aggr_type in ["learnable_weighted", "iterate_learnable_weighted"]:
                self._pre_msg_learnable = True
            else:
                self._pre_msg_learnable = False
                self._processed_feature = self._pre_msg_op.aggregate(self._processed_feat_list)
        else:
            self._pre_msg_learnable = False
            self._processed_feature = feature
        return self._processed_feature

    def postprocess(self, output):
        if self._post_graph_op is not None:
            if self._post_msg_op.aggr_type in ["learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError("Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = self._post_msg_op(self._post_graph_op(output))

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self._pre_msg_learnable is False:
            processed_feature = self._processed_feature[idx].to(device)
        else:
            transferred_feat_list = [feat[idx].to(device) for feat in self._processed_feat_list]
            processed_feature = self._pre_msg_op.aggregate(transferred_feat_list)

        output = self._base_model(processed_feature)
        return output
