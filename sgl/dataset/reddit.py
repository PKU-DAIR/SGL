import numpy as np
import os
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import torch
from torch_geometric.data import extract_zip

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


class Reddit(NodeDataset):
    def __init__(self, name="reddit", root="./", split="official"):
        if name not in ["reddit"]:
            raise ValueError("Dataset name not supported!")
        super(Reddit, self).__init__(root + "Reddit/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
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
        print(url)
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
            data = np.load(osp.join(self._raw_dir, 'reddit_data.npz'))
            split = torch.tensor(data['node_types'])

            train_idx = torch.nonzero(split == 1).reshape(-1)
            val_idx = torch.nonzero(split == 2).reshape(-1)
            test_idx = torch.nonzero(split == 3).reshape(-1)
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
