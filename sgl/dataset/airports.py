import numpy as np
import os.path as osp
import pickle as pkl
import torch

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to, random_split_dataset


class Airports(NodeDataset):
    # The number of nodes in training needs to be carefully selected
    def __init__(self, name="usa", root="./", split="official", num_train_per_class=100, num_valid_per_class=20):
        name = name.lower()
        if name not in ["usa", "brazil", "europe"]:
            raise ValueError("Dataset name not found!")
        super(Airports, self).__init__(root + "Airports/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._num_train_per_class = num_train_per_class
        self._num_valid_per_class = num_valid_per_class
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        return [
            osp.join(self._raw_dir, f'{self._name}-airports.edgelist'),
            osp.join(self._raw_dir, f'labels-{self.name}-airports.txt')
        ]

    @property
    def processed_file_paths(self):
        return osp.join(self._processed_dir, f"{self._name}.graph")

    def _download(self):
        edge_url = ('https://github.com/leoribeiro/struc2vec/'
                    'raw/master/graph/{}-airports.edgelist')
        label_url = ('https://github.com/leoribeiro/struc2vec/'
                     'raw/master/graph/labels-{}-airports.txt')

        edge_url = edge_url.format(self._name)
        label_url = label_url.format(self._name)

        print(edge_url)
        download_to(edge_url, self.raw_file_paths[0])
        print(label_url)
        download_to(label_url, self.raw_file_paths[1])

    def _process(self):
        index_map, ys = {}, []
        with open(self.raw_file_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            for i, row in enumerate(data):
                idx, y = row.split()
                index_map[int(idx)] = i
                ys.append(int(y))
        labels = torch.LongTensor(ys)

        # One-hot features
        features = np.eye(labels.shape[0])
        num_node = features.shape[0]
        node_type = "airport"

        edge_indices = []
        with open(self.raw_file_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            for row in data:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
        edge_index = np.array(edge_indices).T
        row, col = edge_index[0], edge_index[1]

        # Set default edge weight to 1
        edge_weight = np.ones(len(row))
        edge_type = "airport__to__airport"

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            index_map, ys = {}, []
            with open(self.raw_file_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                for i, row in enumerate(data):
                    idx, y = row.split()
                    index_map[int(idx)] = i
                    ys.append(int(y))
            labels = np.array(ys)

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
            train_idx, val_idx, test_idx = random_split_dataset(self.num_node)
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
