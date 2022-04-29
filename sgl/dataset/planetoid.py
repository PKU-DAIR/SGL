import networkx as nx
import numpy as np
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import torch

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


class Planetoid(NodeDataset):
    def __init__(self, name="cora", root="./", split="official"):
        if name not in ["cora", "citeseer", "pubmed"]:
            raise ValueError("Dataset name not supported!")
        super(Planetoid, self).__init__(root + "Planetoid/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(split)

    @property
    def raw_file_paths(self):
        filenames = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [osp.join(self._raw_dir, "ind.{}.{}".format(self._name, filename)) for filename in filenames]

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        url = "https://github.com/kimiyoung/planetoid/raw/master/data"
        for filepath in self.raw_file_paths:
            file_url = url + '/' + osp.basename(filepath)
            print(file_url)
            download_to(file_url, filepath)

    def _normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
            
    def _process(self):
        objects = []
        for raw_file in self.raw_file_paths[:-1]:
            objects.append(pkl_read_file(raw_file))

        x, tx, allx, y, ty, ally, graph = tuple(objects)

        test_idx_reorder = []
        with open(self.raw_file_paths[-1], 'r') as rf:
            try:
                for line in rf:
                    test_idx_reorder.append(int(line.strip()))
            except IOError as e:
                print(e)
                exit(1)
        test_idx_range = np.sort(test_idx_reorder)

        if self._name == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vectors into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = self._normalize(features)
        features = np.array(features.todense())
        num_node = features.shape[0]
        node_type = "paper"

        adj = sp.coo_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)))
        row, col, edge_weight = adj.row, adj.col, adj.data
        edge_type = "paper__to__paper"

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.argmax(labels, 1)
        labels = torch.LongTensor(labels)

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            train_idx = range(self.num_classes * 20)
            val_idx = range(self.num_classes * 20, self.num_classes * 20 + 500)
            test_idx = range(self.num_node - 1000, self.num_node)
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
