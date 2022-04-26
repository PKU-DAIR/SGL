import networkx as nx
import numpy as np
import os.path as osp
import pickle as pkl
import torch

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file


class KarateClub(NodeDataset):
    def __init__(self, name="karateclub", root="./", split="official", num_train_per_class=1, num_valid_per_class=1):
        super(KarateClub, self).__init__(root + "KarateClub/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._num_train_per_class = num_train_per_class
        self._num_valid_per_class = num_valid_per_class
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        return [osp.join(self._raw_dir, "karateclub")]

    @property
    def processed_file_paths(self):
        return osp.join(self._processed_dir, f"{self._name}.graph")

    def _download(self):
        pass

    def _process(self):
        G = nx.karate_club_graph()
        features = np.eye(G.number_of_nodes())
        num_node = features.shape[0]
        node_type = "person"

        if hasattr(nx, 'to_scipy_sparse_array'):
            adj = nx.to_scipy_sparse_array(G).tocoo()
        else:
            adj = nx.to_scipy_sparse_matrix(G).tocoo()

        row, col, edge_weight = adj.row, adj.col, adj.data
        edge_type = "person__to__person"

        # Create communities
        labels = torch.tensor([
            1, 1, 1, 1, 3, 3, 3, 1, 0, 1, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1,
            0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0
        ], dtype=torch.long)

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            labels = np.array([
                1, 1, 1, 1, 3, 3, 3, 1, 0, 1, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1,
                0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0
            ])

            # By Default: Select a single training node (the first one) for each community
            num_train_per_class = self._num_train_per_class
            num_val = self._num_valid_per_class

            train_idx, val_idx, test_idx = np.empty(0), np.empty(0), np.empty(0)
            for i in range(self.num_classes):
                idx = np.nonzero(labels == i)[0]
                train_idx = np.append(train_idx, idx[:num_train_per_class])
                val_idx = np.append(val_idx, idx[num_train_per_class: num_train_per_class + num_val])
                test_idx = np.append(test_idx, idx[num_train_per_class + num_val:])
            train_idx.reshape(-1)
            val_idx.reshape(-1)
            test_idx.reshape(-1)

        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
