import torch
import torch.nn as nn
import torch.nn.functional as F

from sgl.data.base_dataset import HeteroNodeDataset


class BaseSGAPModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseSGAPModel, self).__init__()
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
            if self._pre_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self._pre_msg_learnable = True
            else:
                self._pre_msg_learnable = False
                self._processed_feature = self._pre_msg_op.aggregate(
                    self._processed_feat_list)
        else:
            self._pre_msg_learnable = False
            self._processed_feature = feature

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
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self._pre_msg_learnable is False:
            processed_feature = self._processed_feature[idx].to(device)
        else:
            transferred_feat_list = [feat[idx].to(
                device) for feat in self._processed_feat_list]
            processed_feature = self._pre_msg_op.aggregate(
                transferred_feat_list)

        output = self._base_model(processed_feature)
        return output


class BaseHeteroSGAPModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseHeteroSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim

        self._pre_graph_op, self._pre_msg_op = None, None
        self._aggregator = None
        self._base_model = None

        self._propagated_feat_list_list = None
        self._processed_feature_list = None
        self._pre_msg_learnable = False

    # Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided.
    def preprocess(self, dataset, predict_class,
                   random_subgraph_num=-1, subgraph_edge_type_num=-1,
                   subgraph_list=None):
        if subgraph_list is None and (random_subgraph_num == -1 or subgraph_edge_type_num == -1):
            raise ValueError(
                "Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided!")
        if subgraph_list is not None and (random_subgraph_num != -1 or subgraph_edge_type_num != -1):
            raise ValueError(
                "subgraph_list is provided, random_subgraph_num and subgraph_edge_type_num will be ignored!")

        if not isinstance(dataset, HeteroNodeDataset):
            raise TypeError(
                "Dataset must be an instance of HeteroNodeDataset!")
        elif predict_class not in dataset.node_types:
            raise ValueError("Please input valid node class for prediction!")
        predict_idx = dataset.data.node_id_dict[predict_class]

        if subgraph_list is None:
            subgraph_dict = dataset.nars_preprocess(dataset.edge_types, predict_class,
                                                    random_subgraph_num,
                                                    subgraph_edge_type_num)
            subgraph_list = [(key, subgraph_dict[key])
                             for key in subgraph_dict]

        self._propagated_feat_list_list = [[]
                                           for _ in range(self._prop_steps + 1)]

        for key, value in subgraph_list:
            edge_type_list = []
            for edge_type in key:
                edge_type_list.append(edge_type.split("__")[0])
                edge_type_list.append(edge_type.split("__")[2])
            if predict_class in edge_type_list:
                adj, feature, node_id = value
                propagated_feature = self._pre_graph_op.propagate(adj, feature)

                start_pos = list(node_id).index(predict_idx[0])
                for i, feature in enumerate(propagated_feature):
                    self._propagated_feat_list_list[i].append(
                        feature[start_pos:start_pos + dataset.data.num_node[predict_class]])

    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        feat_input = []
        for x_list in self._propagated_feat_list_list:
            feat_input.append([])
            for x in x_list:
                feat_input[-1].append(x[idx].to(device))

        aggregated_feat_list = self._aggregator(feat_input)
        combined_feat = self._pre_msg_op.aggregate(aggregated_feat_list)
        output = self._base_model(combined_feat)

        return output


class FastBaseHeteroSGAPModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(FastBaseHeteroSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim

        self._pre_graph_op = None
        self._aggregator = None
        self._base_model = None

        self._propagated_feat_list_list = None
        self._processed_feature_list = None
        self._pre_msg_learnable = False

    # Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided.
    def preprocess(self, dataset, predict_class,
                   random_subgraph_num=-1, subgraph_edge_type_num=-1,
                   subgraph_list=None):
        if subgraph_list is None and (random_subgraph_num == -1 or subgraph_edge_type_num == -1):
            raise ValueError(
                "Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided!")
        if subgraph_list is not None and (random_subgraph_num != -1 or subgraph_edge_type_num != -1):
            raise ValueError(
                "subgraph_list is provided, random_subgraph_num and subgraph_edge_type_num will be ignored!")

        if not isinstance(dataset, HeteroNodeDataset):
            raise TypeError(
                "Dataset must be an instance of HeteroNodeDataset!")
        elif predict_class not in dataset.node_types:
            raise ValueError("Please input valid node class for prediction!")
        predict_idx = dataset.data.node_id_dict[predict_class]

        if subgraph_list is None:
            subgraph_dict = dataset.nars_preprocess(dataset.edge_types, predict_class,
                                                    random_subgraph_num,
                                                    subgraph_edge_type_num)
            subgraph_list = [(key, subgraph_dict[key])
                             for key in subgraph_dict]

        self._propagated_feat_list_list = [[]
                                           for _ in range(self._prop_steps + 1)]

        for key, value in subgraph_list:
            edge_type_list = []
            for edge_type in key:
                edge_type_list.append(edge_type.split("__")[0])
                edge_type_list.append(edge_type.split("__")[2])
            if predict_class in edge_type_list:
                adj, feature, node_id = value
                propagated_feature = self._pre_graph_op.propagate(adj, feature)

                start_pos = list(node_id).index(predict_idx[0])
                for i, feature in enumerate(propagated_feature):
                    self._propagated_feat_list_list[i].append(
                        feature[start_pos:start_pos + dataset.data.num_node[predict_class]])

        # 2-d list to 4-d tensor (num_node, feat_dim, num_subgraphs, prop_steps)
        self._propagated_feat_list_list = [torch.stack(
            x, dim=2) for x in self._propagated_feat_list_list]
        self._propagated_feat_list_list = torch.stack(
            self._propagated_feat_list_list, dim=3)

        # 4-d tensor to 3-d tensor (num_node, feat_dim, num_subgraphs * prop_steps)
        shape = self._propagated_feat_list_list.size()
        self._propagated_feat_list_list = self._propagated_feat_list_list.view(
            shape[0], shape[1], shape[2] * shape[3])

    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        feat_input = self._propagated_feat_list_list[idx].to(device)

        aggregated_feat_from_diff_hops = self._aggregator(feat_input)
        output = self._base_model(aggregated_feat_from_diff_hops)

        return output
