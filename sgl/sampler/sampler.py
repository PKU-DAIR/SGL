import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp

from sgl.sampler.base_sampler import BaseSampler

# import metis
import random

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
    
    def sampling(self, batch_inds):
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

        all_adjs = []
        cur_tgt_nodes = batch_inds    
        for layer_index in range(self.num_layers):
            cur_src_nodes, adj_sampled = self._one_layer_sampling(cur_tgt_nodes, self.layer_sizes[layer_index])
            all_adjs.insert(0, adj_sampled)
            cur_tgt_nodes = cur_src_nodes
        
        all_adjs = self._post_process(all_adjs)
     
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

    def sampling(self, batch_inds): 
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
        all_adjs = []

        cur_out_nodes = batch_inds
        for layer_index in range(self.num_layers):
            cur_in_nodes, cur_adj = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_adjs.insert(0, cur_adj)
            cur_out_nodes = cur_in_nodes

        all_adjs = self._post_process(all_adjs)

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
        self._train_idx = dataset.train_idx
        self._val_idx = dataset.val_idx 
        self._test_idx = dataset.test_idx
        self._sampling_done = False

    def _pre_process(self, **kwargs):

        self.cluster_method = kwargs.get("cluster_method", "random")
        self.cluster_number = kwargs.get("cluster_number", 32)

    def sampling(self, cluster_ind, training):
        """
        Decomposing the graph, creating Torch arrays.
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
        
        cluster_ind = cluster_ind.item()
        if training is True:
            batch_out = [self.sg_train_nodes[cluster_ind]]
        else:
            batch_out = [self.sg_val_nodes[cluster_ind], self.sg_test_nodes[cluster_ind]]
        
        return self.sg_nodes[cluster_ind], batch_out, self.sg_edges[cluster_ind]
    
    def _random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = range(self.cluster_number)
        self.cluster_membership = {node: random.choice(self.clusters) for node in self._adj.nodes()}

    # def _metis_clustering(self):
    #     """
    #     Clustering the graph with Metis. For details see:
    #     """
    #     (st, parts) = metis.part_graph(self._adj, self.cluster_number)
    #     self.clusters = list(set(parts))
    #     self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def _general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {cluster: [] for cluster in self.clusters}
        self.sg_val_nodes = {cluster: [] for cluster in self.clusters}
        self.sg_test_nodes = {cluster: [] for cluster in self.clusters}
        for cluster in self.clusters: 
            self.sg_nodes[cluster] = [node for node in sorted(self._adj.nodes()) if self.cluster_membership[node] == cluster]
            subgraph = self._adj.subgraph(self.sg_nodes[cluster]) 
            # map the global node inds to the local node inds
            mapper = {node: i for i, node in enumerate(self.sg_nodes[cluster])}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            for node in self.sg_nodes[cluster]:
                if node in self._train_idx:
                    self.sg_train_nodes[cluster].append([mapper[node], node])
                elif node in self._val_idx:
                    self.sg_val_nodes[cluster].append([mapper[node], node])
                elif node in self._test_idx:
                    self.sg_test_nodes[cluster].append([mapper[node], node])

    def _transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format (except for sg_edges which are coo_matrices currently).
        """
        for cluster in self.clusters:
            num_nodes = len(self.sg_nodes[cluster])
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            row, col = np.array(self.sg_edges[cluster]).transpose()
            self.sg_edges[cluster] = self._post_process(sp.coo_matrix((np.ones(row.shape[0]), (row, col)), 
                                                                      shape=(num_nodes, num_nodes)))
            self.sg_edges[cluster] = self._to_Block(self.sg_edges[cluster])
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster]).transpose_(1, 0)
            self.sg_val_nodes[cluster] = torch.LongTensor(self.sg_val_nodes[cluster]).transpose_(1, 0)
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster]).transpose_(1, 0)