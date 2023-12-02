import os
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm

from sgl.data.base_data import Block
import sgl.operators.graph_op as GraphOps
from sgl.sampler.utils import adj_train_analysis
from sgl.utils import sparse_mx_to_torch_sparse_tensor

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
                adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
        else:
            if self._post_sampling_op is not None:
                adjs = self._post_sampling_op._construct_adj(adjs)
            if to_sparse_tensor:
                adjs = sparse_mx_to_torch_sparse_tensor(adjs)
        return adjs
    
    def _to_Block(self, adjs):
        return Block(adjs)
    
    def collate_fn(self, *args):
        raise NotImplementedError
