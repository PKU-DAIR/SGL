import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

from sgl.sampler.base_sampler import BaseSampler
from sgl.sampler.utils import adj_train_analysis
from sgl.tasks.utils import sparse_mx_to_torch_sparse_tensor

# import metis
import random
from sklearn.model_selection import train_test_split

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
        Neighborhood sampling method follows GraphSAGE.
        Implementation is adapted from PyG.
        """
        super(NeighborSampler, self).__init__(adj, **kwargs)
        self.sampler_name = "NeighborSampler"
        self.pre_sampling = False

    def _preproc(self, **kwargs):
        allowed_kwargs = {"pre_probs", "prob_type", "layer_sizes", "num_layers", "replace", "device"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg
        
        if "layer_sizes" in kwargs.keys():
            if isinstance(kwargs["layer_sizes"], int):
                self.layer_sizes = [kwargs["layer_sizes"]] * kwargs.get("num_layers", 2) # default 2-hop
            else:
                self.layer_sizes = kwargs["layer_sizes"]
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

        self.replace = kwargs.get("replace", False)
        self.device = kwargs.get("device", torch.device("cpu"))
        self.adj_t = self.adj.transpose()

    def sampling(self, batch_inds):
        """
        Intput:
            batch_inds: array of batch node inds
        Method:
            Neighbor sampling
        Outputs:
            n_id: global node index of each node in batch
            adjs: list of sampled adj in the form of 2D tensor [2, M] where M = number of edges
        """
        all_adjs = [[]] * self.num_layers
        cur_tgt_nodes = batch_inds.numpy()
        for layer_index in range(self.num_layers-1, -1, -1):
            cur_src_nodes, adj_sampled = self._one_layer_sampling(cur_tgt_nodes, self.layer_sizes[layer_index])
            all_adjs[layer_index] = adj_sampled
            cur_tgt_nodes = cur_src_nodes
        
        all_adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in all_adjs]
        return {"source_n_ids": cur_tgt_nodes, "sampled_adjs": all_adjs}


    def _one_layer_sampling(self, v_indices, layer_size):
        """
        Inputs:
            v_indices: array of target node inds of the current layer
            layer_size: size of sampled neighbors as the source nodes
        """  
        ret_nodes, ret_edges = [], []
        for v_ind in v_indices: # global id
            st_indptr, ed_indptr = self.adj_t.indptr[v_ind], self.adj_t.indptr[v_ind+1]
            neis = self.adj_t.indices[st_indptr: ed_indptr] # neighbor range         
            p1 = self.probs[neis]
            p1 = p1 / np.sum(p1)
            sample_size = min(ed_indptr-st_indptr, layer_size)
            e_ids = np.random.choice(np.arange(st_indptr, ed_indptr), sample_size, self.replace, p1)
            src_nodes = self.adj_t.indices[e_ids]
            ret_edges.append(e_ids)
            ret_nodes.append(src_nodes)  

        return self._adj_extract(v_indices, ret_nodes, ret_edges)
    
    def _adj_extract(self, tgt_nodes, src_nodes, e_ids):
        row, col, data = [], [], []
        unique_src_nodes = np.unique(np.concatenate(src_nodes))
        # global id to local id
        nid_mapper_tgt = {tgt_nodes[i]: i for i in range(len(tgt_nodes))}
        nid_mapper_src = {unique_src_nodes[i]: i for i in range(len(unique_src_nodes))}
        num_tgt_nodes = len(tgt_nodes)
        for i in range(num_tgt_nodes):
            tgt_node = tgt_nodes[i]
            num_edges = len(e_ids[i])
            col.extend([nid_mapper_tgt[tgt_node]] * num_edges)
            for j in range(num_edges):
                old_ptr = e_ids[i][j]
                src_node = self.adj_t.indices[old_ptr]
                row.append(nid_mapper_src[src_node])
                data.append(self.adj_t[tgt_node, src_node])

        row, col, data = np.array(row), np.array(col), np.array(data)
        adj_sampled = sp.coo_matrix((data, (col, row)), shape=(len(tgt_nodes), len(unique_src_nodes)))

        return unique_src_nodes, adj_sampled


class FastGCNSampler(BaseSampler):
    def __init__(self, adj, **kwargs):   
        super(FastGCNSampler, self).__init__(adj, **kwargs) 
        self.sampler_name = "FastGCNSampler"
        self.pre_sampling = False

    def _preproc(self, **kwargs):
        allowed_kwargs = {"pre_probs", "layer_sizes", "prob_type", "min_neighs", "sim_threshold", "step", "low_quality_score"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg

        self.layer_sizes = kwargs.get("layer_sizes", [1])
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
                raise ValueError("Only support two types of probability calculation: normalize_col and uniform.")
        self.num_layers = len(self.layer_sizes)

    def sampling(self, batch_inds): 
        """
        Input:
            batch_inds: array of batch node inds
        Method:
            Sample fixed size of nodes independently at each layer.
        Outputs:
            cur_out_nodes: array of source node inds at the first layer
            all_support: list of sampled adjs (torch sparse tensor) at each layer
        """
        all_support = [[]] * self.num_layers

        cur_out_nodes = batch_inds
        for layer_index in range(self.num_layers-1, -1, -1):
            cur_in_nodes, cur_support = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_support[layer_index] = cur_support 
            cur_out_nodes = cur_in_nodes

        all_support = [sparse_mx_to_torch_sparse_tensor(adj) for adj in all_support]
        return {"source_n_ids": cur_out_nodes, "sampled_adjs": all_support}

    def _one_layer_sampling(self, v_indices, output_size):
        # NOTE: FastGCN described in paper samples neighboors without reference
        # to the v_indices. But in its tensorflow implementation, it has used
        # the v_indice to filter out the disconnected nodes. So the same thing
        # has been done here.
        """
        Inputs:
            v_indices: array of target node inds of the current layer
            output_size: size of the source nodes to be sampled
        Outputs:
            u_samples: array of source node inds of the current layer
            support: normalized sparse adjacency matrix of the current layer
        """
        # NOTE: Should we transpose adj since v_indices are the target nodes in the process of message propagation?
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        # NOTE: Should sampled contain repeated nodes? 
        sampled = np.random.choice(np.arange(np.size(neis)),
                                   output_size, True, p1) 

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support

    
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

    def _preproc(self, **kwargs):
        allowed_kwargs = {"clustering_method", "cluster_number", "test_ratio"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg
        self.clustering_method = kwargs.get("clustering_method", "random")
        self.cluster_number = kwargs.get("cluster_number", 32)
        self.test_ratio = kwargs.get("test_ratio", 0.3)
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1] 
        self.class_count = np.max(self.target)+1

    def sampling(self, batch_inds):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self._sampling_done is False:
            if self.clustering_method == "metis":
                print("\nMetis graph clustering started.\n")
                # self._metis_clustering()
            else:
                print("\nRandom graph clustering started.\n")
                self._random_clustering()
            self._general_data_partitioning()
            self._transfer_edges_and_nodes()
            self._sampling_done = True
            return {"adj": self.sg_edges, "x": self.sg_features}
        else:
            return {}

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
    #     (st, parts) = metis.part_graph(self.adj, self.cluster_number) # 每个聚类属于哪个part
    #     self.clusters = list(set(parts)) # 一共有几个part
    #     self.cluster_membership = {node: membership for node, membership in enumerate(parts)} # part加入key值，key为节点序号

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
            self.sg_edges[cluster] = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])