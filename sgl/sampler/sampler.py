import os
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from torch_sparse import SparseTensor, cat
from torch_geometric.utils import from_networkx, mask_to_index

from sgl.sampler.base_sampler import BaseSampler

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
        self.full_block = self._to_Block(self._adj)

    def sampling(self):
        return self.full_batch, self.full_batch, self.full_block

class NeighborSampler(BaseSampler):
    def __init__(self, adj, **kwargs):
        """
        Node-wise neighbor sampler
        """
        super(NeighborSampler, self).__init__(adj, **kwargs)
        self.sampler_name = "NeighborSampler"
        self.sample_level = "node"
        self.pre_sampling = False

    def _pre_process(self, **kwargs):

        self._get_sample_sizes(**kwargs)

        self._calc_probs(**kwargs)
        
        self.replace = kwargs.get("replace", True)
    
    def collate_fn(self, batch_inds):
        """
        Input:
            batch_inds: array of batch node inds
        Method:
            Neighbor sampling
        Outputs:
            batch_in: global node index of each source node in the first aggregation layer
            batch_out: global node index of each target node in the last aggregation layer
            block: sampled adjs in the form of sparse tensors wrapped in Block class
        """
        if callable(batch_inds):
            batch_inds = batch_inds()
        if isinstance(batch_inds, torch.Tensor):
            batch_inds = batch_inds.numpy()
        if not isinstance(batch_inds, np.ndarray):
            batch_inds = np.asarray(batch_inds)
        
        all_adjs = []
        cur_tgt_nodes = batch_inds    
        for layer_index in range(self.num_layers):
            cur_src_nodes, adj_sampled = self._one_layer_sampling(cur_tgt_nodes, self.layer_sizes[layer_index])
            all_adjs.insert(0, adj_sampled)
            cur_tgt_nodes = cur_src_nodes
        
        all_adjs = self._post_process(all_adjs, to_sparse_tensor=False)
     
        return cur_tgt_nodes, batch_inds, self._to_Block(all_adjs)  

    def _one_layer_sampling(self, prev_nodes, layer_size=-1):
        """
        Inputs:
            v_indices: array of target node inds of the current layer
            layer_size: size of sampled neighbors as the source nodes
        """  
        
        current_layer_adj = self._adj[prev_nodes, :]

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

class FastGCNSampler(BaseSampler):
    def __init__(self, adj, **kwargs):   
        super(FastGCNSampler, self).__init__(adj, **kwargs) 
        self.sampler_name = "FastGCNSampler"
        self.sample_level = "layer"
        self.pre_sampling = False

    def _pre_process(self, **kwargs):

        self._get_sample_sizes(**kwargs)

        self._calc_probs(**kwargs)

        self.replace = kwargs.get("replace", False)

    def collate_fn(self, batch_inds): 
        """
        Input:
            batch_inds: array of batch node inds
        Method:
            Sample fixed size of nodes independently at each layer.
        Outputs:
            batch_in: global node index of each source node in the first aggregation layer
            batch_out: global node index of each target node in the last aggregation layer
            block: sampled adjs in the form of sparse tensors wrapper in Block class
        """
        if callable(batch_inds):
            batch_inds = batch_inds()
        if not isinstance(batch_inds, np.ndarray):
            batch_inds = np.asarray(batch_inds)
        all_adjs = []

        cur_out_nodes = batch_inds
        for layer_index in range(self.num_layers):
            cur_in_nodes, cur_adj = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_adjs.insert(0, cur_adj)
            cur_out_nodes = cur_in_nodes

        all_adjs = self._post_process(all_adjs, to_sparse_tensor=False)

        return cur_out_nodes, batch_inds, self._to_Block(all_adjs)

    def _one_layer_sampling(self, v_indices, output_size):
        """
        Inputs:
            v_indices: array of target node inds of the current layer
            output_size: size of the source nodes to be sampled
        Outputs:
            u_samples: array of source node inds of the current layer
            support: normalized sparse adjacency matrix of the current layer
        """
        support = self._adj[v_indices, :]
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
    
class ClusterGCNSampler(BaseSampler):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, dataset, **kwargs):
        """
        Inputs:
            adj: Adjacency matrix (Networkx Graph).
        """
        super(ClusterGCNSampler, self).__init__(nx.from_scipy_sparse_matrix(dataset.adj), **kwargs)
        self.sampler_name = "ClusterGCNSampler"
        self.sample_level = "graph"
        self.pre_sampling = True
        self._masks = {"train": dataset.train_mask, "val": dataset.val_mask, "test": dataset.test_mask}
        self._sampling_done = False

    def _pre_process(self, **kwargs):

        self.cluster_method = kwargs.get("cluster_method", "random")
        self.cluster_number = kwargs.get("cluster_number", 32)
        self._save_dir = kwargs.get("save_dir", None)
        if self._save_dir is not None:
            self._save_path_pt = os.path.join(self._save_dir, f"partition_{self.cluster_method}_{self.cluster_number}.pt")
            self._save_path_pkl = os.path.join(self._save_dir, f"partition_{self.cluster_method}_{self.cluster_number}.pkl")
        else:
            self._save_path_pt = self._save_path_pkl = None

    def collate_fn(self, batch_inds, mode):
        if self._sampling_done is False:
            if self._save_dir is not None and os.path.exists(self._save_path_pt) and os.path.exists(self._save_path_pkl):
                print("\nLoad from existing clusters.\n")
                (self.perm_adjs, self.partptr, self.perm_node_idx) = torch.load(self._save_path_pt)
                self.splitted_perm_adjs = pkl.load(open(self._save_path_pkl, "rb"))
            else:
                if self.cluster_method == "metis":
                    print("\nMetis graph clustering started.\n")
                    self._metis_clustering()
                else:
                    raise NotImplementedError
            
            self._sampling_done = True
        
        if not isinstance(batch_inds, torch.Tensor):
            batch_inds = torch.tensor(batch_inds)
        
        # stack len(batch_inds) subgraphs into one graph
        start = self.partptr[batch_inds].tolist()
        end = self.partptr[batch_inds + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
        global_node_idx = self.perm_node_idx[node_idx]
        composed_sparse_mx = sp.block_diag([self.splitted_perm_adjs[batch_ind.item()] for batch_ind in batch_inds])
        block = self._to_Block(composed_sparse_mx)
        if mode in ["train", "val", "test"]:
            mask = self._masks[mode][global_node_idx]
            global_inds = global_node_idx[mask]
            local_inds = mask_to_index(mask)
            batch_out = torch.vstack([local_inds, global_inds])
        else:
            mode = mode.split("_")
            batch_out = {}
            for one_mode in mode:
                mask = self._masks[one_mode][global_node_idx]
                global_inds = global_node_idx[mask]
                local_inds = mask_to_index(mask)
                batch_out.update({one_mode: torch.vstack([local_inds, global_inds])})
        return global_node_idx, batch_out, block

    def _metis_clustering(self):
        data = from_networkx(self._adj)
        N, E = data.num_nodes, data.num_edges
        adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(E, device=data.edge_index.device),
            sparse_sizes=(N, N))
        self.perm_adjs, self.partptr, self.perm_node_idx = adj.partition(self.cluster_number, False) 
        self.splitted_perm_adjs = []
        for i in range(len(self.partptr)-1):
            start, end = self.partptr[i], self.partptr[i+1]
            node_idx = torch.arange(start, end)
            perm_adj = self.perm_adjs.narrow(0, start, end-start)
            perm_adj = perm_adj.index_select(1, node_idx)
            row, col, _ = perm_adj.coo()
            row, col = row.numpy(), col.numpy()
            num_nodes = len(node_idx)
            sparse_mx = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))
            sparse_mx = self._post_process(sparse_mx, to_sparse_tensor=False)
            self.splitted_perm_adjs.append(sparse_mx)
        if self._save_dir is not None:
            torch.save((self.perm_adjs, self.partptr, self.perm_node_idx), self._save_path_pt)
            pkl.dump(self.splitted_perm_adjs, open(self._save_path_pkl, "wb"))
            print(f"\nSave Metis graph clustering results under the {self._save_dir} directory.\n")