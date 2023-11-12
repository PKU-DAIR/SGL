import torch
import torch.nn as nn
import torch.nn.functional as F

from sgl.data.base_dataset import HeteroNodeDataset
from sgl.tasks.utils import sparse_mx_to_torch_sparse_tensor


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

        output = self._base_model(processed_feature) # model training
        return output

class BaseSAMPLEModel(nn.Module):
    def __init__(self, evaluate_mode="full"):
        super(BaseSAMPLEModel, self).__init__()

        self._pre_graph_op, self._post_graph_op = None, None
        self._sampling_op, self._post_sampling_graph_op = None, None
        self._base_model = None
        self._evaluate_mode = evaluate_mode

        self._processed_feat_list = None
        self._processed_feature = None
        self._pre_msg_learnable = False
        self._norm_adj = None

    @property
    def pre_sampling(self):
        return self._sampling_op.pre_sampling
    
    @property
    def sampler_name(self):
        return self._sampling_op.sampler_name
    
    @property
    def evaluate_mode(self):
        return self._evaluate_mode

    def sampling(self, batch_inds, to_sparse_tensor=True):
        sample_results = self._sampling_op.sampling(batch_inds)
        adjs = sample_results.get("sampled_adjs", None)
        if adjs is not None:
            if isinstance(adjs, list):
                if self._post_sampling_graph_op is not None:
                    adjs = [self._post_sampling_graph_op._construct_adj(adj) for adj in adjs]
                if to_sparse_tensor:
                    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
            elif isinstance(adjs, dict):
                if self._post_sampling_graph_op is not None:
                    adjs = {sg_id: self._post_sampling_graph_op._construct_adj(adj) for sg_id, adj in adjs.items()}
                if to_sparse_tensor:
                    adjs = {sg_id: sparse_mx_to_torch_sparse_tensor(adj) for sg_id, adj in adjs.items()}
            else:
                if self._post_sampling_graph_op is not None:
                    adjs = self._post_sampling_graph_op._construct_adj(adjs)
                if to_sparse_tensor:
                    adjs = sparse_mx_to_torch_sparse_tensor(adjs)
            sample_results.update({"sampled_adjs": adjs})
        return sample_results  
       
    def preprocess(self, adj, x):
        if self._pre_graph_op is not None:
            # We don't transform _norm_adj into the form of sparse tensor, 
            # as sparse tensors don't have strides.
            self._norm_adj = self._pre_graph_op._construct_adj(adj)
        else:
            # For ClusterGCN, we have already processed subgraphs after sampling.
            self._norm_adj = adj
        self._pre_msg_learnable = False
        if hasattr(self, "_pre_feature_op"):
            self._processed_feature = self._pre_feature_op._transform_x(x)
        else:
            self._processed_feature = x
    
    def postprocess(self, adj, output):
        if self._post_graph_op is not None:
            raise NotImplementedError
        return output
    
    # a wrapper of the forward function
    def model_forward(self, batch_idx, device, **kwargs):
        return self.forward(batch_idx, device, **kwargs)

    def forward(self, batch_idx, device, **kwargs):
        sampler_name = self._sampling_op.sampler_name    
        if self.training: 
            if sampler_name in ["FastGCNSampler", "NeighborSampler"]:
                sampled_adjs = kwargs["sampled_adjs"]
                n_ids = kwargs["n_ids"]
                sampled_x = self._processed_feature[n_ids].to(device)
                sampled_adjs = [sampled_adj.to(device) for sampled_adj in sampled_adjs]
                effective_batch = batch_idx
                output = self._base_model(sampled_x, sampled_adjs)
            elif sampler_name == "ClusterGCNSampler":
                batch_idx = batch_idx.item()
                sampled_x = self._processed_feature[batch_idx].to(device)
                sampled_adj = self._norm_adj[batch_idx].to(device)
                effective_batch = self._sampling_op.sg_train_nodes[batch_idx]
                output = self._base_model(sampled_x, sampled_adj)[effective_batch]
            elif sampler_name == "FullSampler":
                full_x = self._processed_feature.to(device)
                full_adj = sparse_mx_to_torch_sparse_tensor(self._norm_adj).to(device)
                output = self._base_model(full_x, full_adj)[batch_idx]
                return output
            else:
                raise ValueError(f"{sampler_name} hasn't been implemented yet!")
        else:
            if sampler_name in ["FastGCNSampler", "NeighborSampler"]:
                full_x = self._processed_feature.to(device)
                num_layers = self._sampling_op.num_layers
                sampled_adjs = [sparse_mx_to_torch_sparse_tensor(self._norm_adj).to(device)] * (num_layers - 1)
                sampled_adjs.append(sparse_mx_to_torch_sparse_tensor(self._norm_adj[batch_idx, :]).to(device))
                effective_batch = batch_idx
                output = self._base_model(full_x, sampled_adjs, tgt_nids=batch_idx)
            elif sampler_name == "ClusterGCNSampler":
                batch_idx = batch_idx.item()
                sampled_x = self._processed_feature[batch_idx].to(device)
                sampled_adj = self._norm_adj[batch_idx].to(device)
                effective_batch = self._sampling_op.sg_test_nodes[batch_idx]
                output = self._base_model(sampled_x, sampled_adj)[effective_batch]
            elif sampler_name == "FullSampler":
                full_x = self._processed_feature.to(device)
                full_adj = sparse_mx_to_torch_sparse_tensor(self._norm_adj).to(device)
                output = self._base_model(full_x, full_adj)[batch_idx]
                return output
            else:
                raise ValueError(f"{sampler_name} hasn't been implemented yet!")
        return output, effective_batch
    
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
