from fileinput import filename
import os.path as osp
import pickle as pkl
import os

import networkx as nx
import numpy as np
from pydantic import FilePath
import scipy.sparse as sp
import torch

from dataset.base_data import Graph
from dataset.base_dataset import NodeDataset
from dataset.utils import pkl_read_file, download_to
from torch_geometric.data import extract_zip


class Reddit(NodeDataset):
    def __init__(self, name="reddit", root="./", split="official"):
        if name not in ["reddit"]:
            raise ValueError("Dataset name not supported!")
        super(Reddit, self).__init__(root + "Reddit/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._train_mask, self._val_mask, self._test_mask = None, None, None
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        filenames = ["reddit_data.npz", "reddit_graph.npz"]
        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        url = "https://data.dgl.ai/dataset/reddit.zip"
        path = osp.join(self._raw_dir, "reddit.zip")
        download_to(url, path)
        extract_zip(path, self._raw_dir)
        os.unlink(path)

    def _process(self):
        adj = sp.coo_matrix(sp.load_npz(
            osp.join(self._raw_dir, 'reddit_graph.npz')))
        row, col, edge_weight = adj.row, adj.col, adj.data
        edge_type = "post__to__post"

        data = np.load(osp.join(self._raw_dir, 'reddit_data.npz'))
        features, labels, split = data['feature'], data['label'], data['node_types']
        num_node = features.shape[0]
        node_type = "post"
        labels = torch.LongTensor(labels)

        self._train_mask = split == 1
        self._val_mask = split == 2
        self._test_mask = split == 3

        g = Graph(row, col, edge_weight, num_node,
                  node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            # Reload mask when _process() is not Called
            if self._train_mask == None:
                data = np.load(osp.join(self._raw_dir, 'reddit_data.npz'))
                split = data['node_types']
                self._train_mask = split == 1
                self._val_mask = split == 2
                self._test_mask = split == 3

            train_idx = np.where(self._train_mask == True)
            val_idx = np.where(self._val_mask == True)
            test_idx = np.where(self._test_mask == True)
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx

# for test
# dataset = Reddit()
