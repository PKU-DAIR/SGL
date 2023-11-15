import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

import sgl.operators.graph_op as GraphOps
from sgl.sampler.base_sampler import BaseSampler
from sgl.sampler.utils import adj_train_analysis
from sgl.utils import sparse_mx_to_torch_sparse_tensor

# import metis
import random
from sklearn.model_selection import train_test_split

LOCALITY_KWARGS = {"min_neighs", "sim_threshold", "step", "low_quality_score"}
UNI_KWARGS = {"pre_sampling_op", "post_sampling_op"}

class FullSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        """
        In fact, this sampler simply returns the full graph.
        """
        super(FullSampler, self).__init__(adj, **kwargs)
        self.sampler_name = "FullSampler"
        self.pre_sampling = False

    def sampling(self, batch_inds):
        return {}

class NeighborSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        """
        Node-wise neighbor sampler
        """
        super(NeighborSampler, self).__init__(adj, **kwargs)
        self.sampler_name = "NeighborSampler"
        self.pre_sampling = False

    def _pre_process(self, **kwargs):
        specific_kwargs = {"pre_probs", "prob_type", "layer_sizes", "num_layers", "replace"}
        for kwarg in kwargs.keys():
            assert kwarg in specific_kwargs or kwarg in LOCALITY_KWARGS or kwarg in UNI_KWARGS, "Invalid keyword argument: " + kwarg
        
        if "pre_sampling_op" in kwargs.keys():
            if kwargs["pre_sampling_op"] == "LaplacianGraphOp":
                graph_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5, add_self_loops=False)
            elif kwargs["pre_sampling_op"] == "RwGraphOp":   
                graph_op = getattr(GraphOps, "RwGraphOp")()
            self.adj = graph_op._construct_adj(self.adj)

        if "post_sampling_op" in kwargs.keys():
            if kwargs["post_sampling_op"] == "LaplacianGraphOp":
                self._post_sampling_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5, add_self_loops=False)
            elif kwargs["post_sampling_op"] == "RwGraphOp":
                self._post_sampling_op = getattr(GraphOps, "RwGraphOp")()

        if "layer_sizes" in kwargs.keys():
            layer_sizes = kwargs["layer_sizes"].split(",")
            layer_sizes = [int(layer_size) for layer_size in layer_sizes]
            self.layer_sizes = layer_sizes
        else:
            raise ValueError("Please provide layer sizes in the form of either a list or an integer!")
        self.num_layers = len(self.layer_sizes)

        if "pre_probs" in kwargs.keys():
            self.probs = kwargs["pre_probs"]
        else:
            prob_type = kwargs.get("prob_type", "normalize")
            if prob_type == "normalize":
                col_norm = sparse_norm(self.adj, axis=0)
                self.probs = col_norm / np.sum(col_norm)
            elif prob_type == "uniform":
                self.probs = np.ones(self.adj.shape[1])
            elif prob_type == "locality":
                """
                This sampling strategy refers to GNNSampler [https://github.com/ICT-GIMLab/GNNSampler]
                """
                min_neighs = kwargs.get("min_neighs", 2)
                sim_threshold = kwargs.get("sim_threshold", 0.1)
                step = kwargs.get("step", 1)
                low_quality_score = kwargs.get("low_quality_score", 0.1)
                locality_score = adj_train_analysis(self.adj, min_neighs, sim_threshold, step, low_quality_score)
                self.probs = locality_score / np.sum(locality_score)
            else:
                raise ValueError(f"Don\'t support {prob_type} probability calculation. "
                                 "Consider pre-calculating the probability and transfer it to pre_probs.")
        
        self.replace = kwargs.get("replace", True)
        # When layer_size = -1, NeighborSampler always returns the same subgraph given the same batch_inds.
        # So we can cache the subgraphs to save the time.
    
    def sampling(self, batch_inds):
        """
        Input:
            batch_inds: array of batch node inds
        Method:
            Neighbor sampling
        Outputs:
            n_id: global node index of each node in batch
            adjs: list of sampled adj in the form of sparse tensors
        """
        if callable(batch_inds):
            batch_inds = batch_inds()

        if isinstance(batch_inds, torch.Tensor):
            batch_inds = batch_inds.numpy()

        all_adjs = []
        cur_tgt_nodes = batch_inds    
        for layer_index in range(self.num_layers):
            cur_src_nodes, adj_sampled = self._one_layer_sampling(cur_tgt_nodes, self.layer_sizes[layer_index])
            all_adjs.append(adj_sampled)
            cur_tgt_nodes = cur_src_nodes
        
        all_adjs = self._post_process(all_adjs[::-1])
     
        return {"batch_in": cur_tgt_nodes, "batch_out": batch_inds, "sampled_adjs": all_adjs}   

    def _one_layer_sampling(self, prev_nodes, layer_size=-1):
        """
        Inputs:
            v_indices: array of target node inds of the current layer
            layer_size: size of sampled neighbors as the source nodes
        """  
        current_layer_adj = self.adj[prev_nodes, :]

        if layer_size == -1:
            # in case layer_size == -1, we simply keep all the neighbors
            next_nodes = np.unique(current_layer_adj.indices)
            
        else:
            next_nodes = []

            row_start_stop = np.lib.stride_tricks.as_strided(current_layer_adj.indptr, shape=(current_layer_adj.shape[0], 2), strides=2*current_layer_adj.indptr.strides)

            for start, stop in row_start_stop:
                neigh_index = current_layer_adj.indices[start:stop]
                probs = self.probs[neigh_index] / np.sum(self.probs[neigh_index])
                num_samples = np.min([neigh_index.size, layer_size]) if self.replace is False else layer_size
                sampled_nodes = np.random.choice(neigh_index, num_samples, replace=self.replace, p=probs)
                next_nodes.append(sampled_nodes)
            
            next_nodes = np.unique(np.concatenate(next_nodes))
        
        next_nodes = np.setdiff1d(next_nodes, prev_nodes)
        next_nodes = np.concatenate((prev_nodes, next_nodes))
        
        return next_nodes, current_layer_adj[:, next_nodes]

    def _post_process(self, adjs, to_sparse_tensor=True):
        if self._post_sampling_op is not None:
            adjs = [self._post_sampling_op._construct_adj(adj) for adj in adjs]
        if to_sparse_tensor:
            adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
        return adjs

class FastGCNSampler(BaseSampler):
    def __init__(self, adj, **kwargs):   
        super(FastGCNSampler, self).__init__(adj, **kwargs) 
        self.sampler_name = "FastGCNSampler"
        self.pre_sampling = False

    def _pre_process(self, **kwargs):
        specific_kwargs = {"pre_probs", "prob_type", "layer_sizes", "replace", "pre_sampling_op"}
        for kwarg in kwargs.keys():
            assert kwarg in specific_kwargs or kwarg in LOCALITY_KWARGS or kwarg in UNI_KWARGS, "Invalid keyword argument: " + kwarg

        if "pre_sampling_op" in kwargs.keys():
            if kwargs["pre_sampling_op"] == "LaplacianGraphOp":
                graph_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5, add_self_loops=False)
            elif kwargs["pre_sampling_op"] == "RwGraphOp":   
                graph_op = getattr(GraphOps, "RwGraphOp")()
            self.adj = graph_op._construct_adj(self.adj)

        if "post_sampling_op" in kwargs.keys():
            if kwargs["post_sampling_op"] == "LaplacianGraphOp":
                self._post_sampling_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5, add_self_loops=False)
            elif kwargs["post_sampling_op"] == "RwGraphOp":
                self._post_sampling_op = getattr(GraphOps, "RwGraphOp")()

        if "layer_sizes" in kwargs.keys():
            layer_sizes = kwargs["layer_sizes"].split(",")
            layer_sizes = [int(layer_size) for layer_size in layer_sizes]
            self.layer_sizes = layer_sizes
        else:
            raise ValueError("Please provide layer sizes in the form of either a list or an integer!")
        self.num_layers = len(self.layer_sizes)

        if "pre_probs" in kwargs.keys():
            self.probs = kwargs["pre_probs"]
        else:
            prob_type = kwargs.get("prob_type", "normalize")
            if prob_type == "normalize":
                col_norm = sparse_norm(self.adj, axis=0)
                self.probs = col_norm / np.sum(col_norm)
            elif prob_type == "uniform":
                self.probs = np.ones(self.adj.shape[1])
            elif prob_type == "locality":
                """
                This sampling strategy refers to GNNSampler [https://github.com/ICT-GIMLab/GNNSampler]
                """
                min_neighs = kwargs.get("min_neighs", 2)
                sim_threshold = kwargs.get("sim_threshold", 0.1)
                step = kwargs.get("step", 1)
                low_quality_score = kwargs.get("low_quality_score", 0.1)
                locality_score = adj_train_analysis(self.adj, min_neighs, sim_threshold, step, low_quality_score)
                self.probs = locality_score / np.sum(locality_score)
            else:
                raise ValueError(f"Don\'t support {prob_type} probability calculation. "
                                 "Consider pre-calculating the probability and transfer it to pre_probs.")
        self.replace = kwargs.get("replace", False)

    def sampling(self, batch_inds): 
        """
        Input:
            batch_inds: array of batch node inds
        Method:
            Sample fixed size of nodes independently at each layer.
        Outputs:
            cur_out_nodes: array of source node inds at the first layer
            all_adjs list of sampled adjs (torch sparse tensor) at each layer
        """
        all_adjs = []

        cur_out_nodes = batch_inds
        for layer_index in range(self.num_layers):
            cur_in_nodes, cur_adj = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_adjs.append(cur_adj)
            cur_out_nodes = cur_in_nodes

        all_adjs = self._post_process(all_adjs[::-1])

        return {"batch_in": cur_out_nodes, "batch_out": batch_inds, "sampled_adjs": all_adjs}

    def _one_layer_sampling(self, v_indices, output_size):
        """
        Inputs:
            v_indices: array of target node inds of the current layer
            output_size: size of the source nodes to be sampled
        Outputs:
            u_samples: array of source node inds of the current layer
            support: normalized sparse adjacency matrix of the current layer
        """
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        if self.replace is False:
            output_size = min(len(neis), output_size)
        sampled = np.random.choice(np.arange(np.size(neis)),
                                   output_size, self.replace, p1) 

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support

    def _post_process(self, adjs, to_sparse_tensor=True):
        if self._post_sampling_op is not None:
            adjs = [self._post_sampling_op._construct_adj(adj) for adj in adjs]
        if to_sparse_tensor:
            adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
        
        return adjs
    
class ClusterGCNSampler(BaseSampler):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, adj, features, target, **kwargs):
        """
        Inputs:
            adj: Adjacency matrix (Networkx Graph).
            features: Feature matrix (ndarray).
            target: Target vector (ndarray).
        """
        self.features = features
        self.target = target
        super(ClusterGCNSampler, self).__init__(adj, **kwargs)
        self.sampler_name = "ClusterGCNSampler"
        self.pre_sampling = True
        self._sampling_done = False

    def _pre_process(self, **kwargs):
        specific_kwargs = {"cluster_method", "cluster_number", "test_ratio"}
        for kwarg in kwargs.keys():
            assert kwarg in specific_kwargs or kwarg in UNI_KWARGS, "Invalid keyword argument: " + kwarg

        if "post_sampling_op" in kwargs.keys():
            if kwargs["post_sampling_op"] == "LaplacianGraphOp":
                self._post_sampling_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5)
            elif kwargs["post_sampling_op"] == "RwGraphOp":
                self._post_sampling_op = getattr(GraphOps, "RwGraphOp")()
        self.cluster_method = kwargs.get("cluster_method", "random")
        self.cluster_number = kwargs.get("cluster_number", 32)
        self.test_ratio = kwargs.get("test_ratio", 0.3)
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1] 
        self.class_count = np.max(self.target)+1

    def sampling(self, batch_inds, training):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self._sampling_done is False:
            if self.cluster_method == "metis":
                print("\nMetis graph clustering started.\n")
                # self._metis_clustering()
            else:
                print("\nRandom graph clustering started.\n")
                self._random_clustering()
            self._general_data_partitioning()
            self._transfer_edges_and_nodes()
            
            self._sampling_done = True
        
        batch_inds = batch_inds.item()
        effective_batch = self.sg_train_nodes[batch_inds] if training else self.sg_test_nodes[batch_inds]
        return {"adj": self.sg_edges[batch_inds], "x": self.sg_features[batch_inds], "effective_batch": effective_batch}

    def _post_process(self, adj, to_sparse_tensor=True):
        if self._post_sampling_op is not None:
            adj = self._post_sampling_op._construct_adj(adj)
        if to_sparse_tensor:
            adj = sparse_mx_to_torch_sparse_tensor(adj)
        return adj
    
    def _random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = range(self.cluster_number)
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.adj.nodes()}

    # def _metis_clustering(self):
    #     """
    #     Clustering the graph with Metis. For details see:
    #     """
    #     (st, parts) = metis.part_graph(self.adj, self.cluster_number)
    #     self.clusters = list(set(parts))
    #     self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def _general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        for cluster in self.clusters: 
            # split train/test within each cluster
            subgraph = self.adj.subgraph([node for node in sorted(self.adj.nodes()) if self.cluster_membership[node] == cluster]) 
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            # map the global node inds to the local node inds
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.test_ratio)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]

    def _transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format (except for sg_edges which are coo_matrices currently).
        """
        for cluster in self.clusters:
            num_nodes = len(self.sg_nodes[cluster])
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            row, col = np.array(self.sg_edges[cluster]).transpose()
            self.sg_edges[cluster] = self._post_process(sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes)))
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])