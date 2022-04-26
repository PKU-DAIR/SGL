import numpy as np
import os.path as osp
import pickle as pkl
import torch

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to, random_split_dataset


class Facebook(NodeDataset):
    def __init__(self, name="facebook", root="./", split="official", num_train_per_class=30, num_valid_per_class=100):
        super(Facebook, self).__init__(root + "Facebook/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._num_train_per_class = num_train_per_class
        self._num_valid_per_class = num_valid_per_class
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(split)

    @property
    def raw_file_paths(self):
        filenames = ["facebook.npz"]
        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        url = "https://graphmining.ai/datasets/ptg/facebook.npz"
        path = osp.join(self._raw_dir, "facebook.npz")
        print(url)
        download_to(url, path)

    def _process(self):
        data = np.load(self.raw_file_paths[0])
        features = data["features"]
        num_node = features.shape[0]
        node_type = "page"

        labels = data["target"]
        labels = torch.LongTensor(labels)

        edge_index = data["edges"].T
        row, col = edge_index[0], edge_index[1]
        edge_type = "page__to__page"

        # Default Edge weights
        edge_weight = np.ones(len(row))

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            data = np.load(self.raw_file_paths[0])
            labels = data["target"]

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
