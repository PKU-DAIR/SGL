import os
import torch
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

from sgl.data.base_data import Block
import sgl.operators.graph_op as GraphOps
from sgl.sampler.utils import adj_train_analysis
from sgl.utils import sparse_mx_to_torch_sparse_tensor, sparse_mx_to_pyg_sparse_tensor

from sampling_ops import NodeWiseOneLayer

SPARSE_TRANSFORM = {"pyg": sparse_mx_to_pyg_sparse_tensor, "torch": sparse_mx_to_torch_sparse_tensor}

class BaseSampler:
    def __init__(self, adj, **kwargs):
        self._adj = adj
        self.sampler_name = "None"
        self.sample_level = "None"
        self._post_sampling_op = None
        self.pre_sampling = False

        if "pre_sampling_op" in kwargs.keys():
            graph_op = kwargs.pop("pre_sampling_op")
            if graph_op == "LaplacianGraphOp":
                graph_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5)
            elif graph_op == "RwGraphOp":   
                graph_op = getattr(GraphOps, "RwGraphOp")()
            self._adj = graph_op._construct_adj(self._adj)
        
        if "post_sampling_op" in kwargs.keys():
            graph_op = kwargs.pop("post_sampling_op")
            if graph_op == "LaplacianGraphOp":
                self._post_sampling_op = getattr(GraphOps, "LaplacianGraphOp")(r=0.5)
            elif graph_op == "RwGraphOp":
                self._post_sampling_op = getattr(GraphOps, "RwGraphOp")()

        self._sparse_type = kwargs.get("sparse_type", "pyg")

        self._pre_process(**kwargs)

    def _pre_process(self, **kwargs):
        pass

    def _get_sample_sizes(self, **kwargs):
        if "layer_sizes" in kwargs.keys():
            layer_sizes = kwargs.pop("layer_sizes").split(",")
            layer_sizes = [int(layer_size) for layer_size in layer_sizes]
            self.layer_sizes = layer_sizes
        else:
            raise ValueError("Please provide layer sizes in the form of either a list or an integer!")
        self.num_layers = len(self.layer_sizes)

    def _calc_probs(self, **kwargs):
        prob_type = kwargs.get("prob_type", "normalize")
        save_dir = kwargs.get("save_dir", None)
        if save_dir is not None:
            pre_calc_path = os.path.join(save_dir, f"{prob_type}_sample_probs.npy")
            if os.path.exists(pre_calc_path):
                self.probs = np.load(pre_calc_path)
                print(f"Load from pre-calculated sampling probability from {str(pre_calc_path)}.")
                return
        if prob_type == "normalize":
            col_norm = sparse_norm(self._adj, axis=0)
            self.probs = col_norm / np.sum(col_norm)
        elif prob_type == "uniform":
            self.probs = np.ones(self._adj.shape[1])
        elif prob_type == "locality":
            """
            This sampling strategy refers to GNNSampler [https://github.com/ICT-GIMLab/GNNSampler]
            """
            min_neighs = kwargs.get("min_neighs", 2)
            sim_threshold = kwargs.get("sim_threshold", 0.1)
            step = kwargs.get("step", 1)
            low_quality_score = kwargs.get("low_quality_score", 0.1)
            locality_score = adj_train_analysis(self._adj, min_neighs, sim_threshold, step, low_quality_score)
            self.probs = locality_score / np.sum(locality_score)
        else:
            raise ValueError(f"Don\'t support {prob_type} probability calculation. "
                                "Consider pre-calculating the probability and transfer it to pre_probs.")
        if save_dir is not None:
            np.save(open(pre_calc_path, "wb"), self.probs)
            print(f"Save the sampling probability into {str(pre_calc_path)}.")
    
    def _post_process(self, adjs, to_sparse_tensor=True):
        if isinstance(adjs, list):
            if self._post_sampling_op is not None:
                adjs = [self._post_sampling_op._construct_adj(adj) for adj in adjs]
            if to_sparse_tensor:
                sparse_transform_func = SPARSE_TRANSFORM.get(self._sparse_type)
                adjs = [sparse_transform_func(adj) for adj in adjs]
        else:
            if self._post_sampling_op is not None:
                adjs = self._post_sampling_op._construct_adj(adjs)
            if to_sparse_tensor:
                sparse_transform_func = SPARSE_TRANSFORM.get(self._sparse_type)
                adjs = [sparse_transform_func(adj) for adj in adjs]
        return adjs
    
    def collate_fn(self, *args):
        raise NotImplementedError

class FullSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        """
        In fact, this sampler simply returns the full graph.
        """
        super(FullSampler, self).__init__(adj, **kwargs)
        self.sampler_name = "FullSampler"
        self.sample_level = "graph"
        self.pre_sampling = False
        self.full_batch = kwargs.get("node_ids", range(self._adj.shape[0]))
        self.full_block = Block(self._adj, self._sparse_type)

    def sampling(self):
        return self.full_batch, self.full_batch, self.full_block
 
class NodeWiseSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        super(NodeWiseSampler, self).__init__(adj, **kwargs)
        self.__indptr = self._adj.indptr
        self.__indices = self._adj.indices
        self.__values = self._adj.data

    def _pre_process(self, **kwargs):
        self._get_sample_sizes(**kwargs)
        self._calc_probs(**kwargs)     
        self.replace = kwargs.get("replace", True)

    def one_layer_sampling(self, target_nodes, layer_size, biased):        
        source_nodes, (s_indptr, s_indices, s_data) = NodeWiseOneLayer(target_nodes, self.__indptr, self.__indices, self.__values, layer_size, self.probs, biased, self.replace)
        subgraph_adj = sp.csr_matrix((s_data, s_indices, s_indptr), shape=(len(target_nodes), len(source_nodes)))
        
        return source_nodes, subgraph_adj
    
class LayerWiseSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        super(LayerWiseSampler, self).__init__(adj, **kwargs)

    def _pre_process(self, **kwargs):
        self._get_sample_sizes(**kwargs)
        self._calc_probs(**kwargs)
        self.replace = kwargs.get("replace", False)

    def one_layer_sampling(self, target_nodes, layer_size, probability):
        subgraph_adj = self._adj[target_nodes, :]
        neis = np.nonzero(np.sum(subgraph_adj, axis=0))[1]
        p1 = probability[neis]
        p1 = p1 / np.sum(p1)

        if self.replace is False:
            layer_size = min(len(neis), layer_size)
        
        local_nids = np.random.choice(np.arange(np.size(neis)),
                                   layer_size, self.replace, p1)
        
        source_nodes = neis[local_nids]
        subgraph_adj = subgraph_adj[:, source_nodes]
        sampled_p1 = p1[local_nids]

        subgraph_adj = subgraph_adj.dot(sp.diags(1.0 / (sampled_p1 * layer_size)))
        return source_nodes, subgraph_adj
    
class GraphWiseSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        super(GraphWiseSampler, self).__init__(adj, **kwargs)

    @property
    def sample_graph_ops(self):
        # Each subclass must implement its own sample operations
        raise NotImplementedError
    
    def multiple_graphs_sampling(self):
        if self.pre_sampling is False or self.sampling_done is False:
            if self._save_dir is not None and os.path.exists(self._save_path_pt) and os.path.exists(self._save_path_pkl):
                print("\nLoad from existing subgraphs.\n")
                (self.perm_adjs, self.partptr, self.perm_node_idx) = torch.load(self._save_path_pt)
                self.splitted_perm_adjs = pkl.load(open(self._save_path_pkl, "rb"))
            else:
                self.sample_graph_ops()        
            self.sampling_done = True    
        else:
            print("\nSubgraphs already existed.\n")