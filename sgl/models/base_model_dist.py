import torch
import torch.nn as nn
import torch.nn.functional as F

from sgl.data.base_dataset import HeteroNodeDataset


class BaseSGAPModelDist(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseSGAPModelDist, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim

        self._pre_graph_op, self._pre_msg_op = None, None
        self._post_graph_op, self._post_msg_op = None, None
        self._base_model = None

        self._processed_feat_list = None
        self._processed_feature = None
        self._pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self._pre_graph_op is not None:
            self._processed_feat_list = self._pre_graph_op.propagate(
                adj, feature)
        else:
            self._processed_feat_list = [feature]

    def postprocess(self, adj, output):
        if self._post_graph_op is not None:
            if self._post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self._post_graph_op.propagate(adj, output)
            output = self._post_msg_op.aggregate(output)

        return output

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
