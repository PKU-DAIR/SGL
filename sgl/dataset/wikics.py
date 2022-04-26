import json
import numpy as np
import os.path as osp
import pickle as pkl
import torch
from itertools import chain

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to, to_undirected


class Wikics(NodeDataset):
    # Have 20 different split of training and validation set, identified by split_id in [0, 19]
    # Currently, we only support calculating the accuracy of one split, 
    # and average accuracy will be supported in the future.
    def __init__(self, name="wikics", root="./", split="official", is_undirected=True, split_id=0):
        if split_id not in range(20):
            raise ValueError("Split id not supported")

        self._is_undirected = is_undirected
        super(Wikics, self).__init__(root + "Wikics", name)

        self._split_id = split_id
        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(split)

    @property
    def raw_file_paths(self):
        filenames = ["data.json"]
        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        name = self._name + "_undirected" if self._is_undirected else self._name
        return osp.join(self._processed_dir, f"{name}.graph")

    def _download(self):
        url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'
        path = self.raw_file_paths[0]
        file_url = url + 'data.json'
        print(file_url)
        download_to(file_url, path)

    def _process(self):
        with open(self.raw_file_paths[0], 'r') as f:
            data = json.load(f)

        features = np.array(data['features'])
        labels = torch.LongTensor(data['labels'])
        num_node = features.shape[0]
        node_type = "article"

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if self._is_undirected:
            edge_index = to_undirected(edge_index)

        edge_index = edge_index.numpy()
        row, col = edge_index[0], edge_index[1]
        edge_weight = np.ones(len(row))
        edge_type = "article__to__article"

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            with open(self.raw_file_paths[0], 'r') as f:
                data = json.load(f)

            train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
            train_mask = train_mask.contiguous()

            val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
            val_mask = val_mask.contiguous()

            stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
            stopping_mask = stopping_mask.contiguous()

            test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

            # Choose the specific split
            train_mask = train_mask[self._split_id]
            val_mask = val_mask[self._split_id]
            stopping_mask = stopping_mask[self._split_id]

            # Different from PyG, we merged the stopping_idx into valid_idx
            train_idx = torch.nonzero(train_mask == 1).reshape(-1)
            val_idx = torch.nonzero(val_mask == 1).reshape(-1)
            stopping_idx = torch.nonzero(stopping_mask == 1).reshape(-1)
            val_idx = torch.cat((val_idx, stopping_idx), dim=0)
            test_idx = torch.nonzero(test_mask == 1).reshape(-1)

        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
