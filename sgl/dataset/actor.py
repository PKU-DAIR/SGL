import numpy as np
import os.path as osp
import pickle as pkl
import torch
from torch_sparse import SparseTensor, coalesce

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


class Actor(NodeDataset):
    # Have 10 different split of training and validation set, identified by split_id in [0, 9]
    # Currently, we only support calculating the accuracy of one split, 
    # and average accuracy will be supported in the future.
    def __init__(self, name="actor", root="./", split="official", split_id=0):
        if split_id not in range(10):
            raise ValueError("Split id not supported")
        super(Actor, self).__init__(root + "Actor", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split, self._split_id = split, split_id
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        filenames = ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                     ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]
        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        return osp.join(self._processed_dir, f"{self._name}.graph")

    def _download(self):
        url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

        for raw_file_path in self.raw_file_paths[:2]:
            raw_file_name = osp.basename(raw_file_path)
            file_url = f'{url}/new_data/film/{raw_file_name}'
            print(file_url)
            download_to(file_url, raw_file_path)

        for raw_file_path in self.raw_file_paths[2:]:
            raw_file_name = osp.basename(raw_file_path)
            file_url = f'{url}/splits/{raw_file_name}'
            print(file_url)
            download_to(file_url, raw_file_path)

    def _process(self):
        with open(self.raw_file_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            labels = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                labels[int(n_id)] = int(label)

        features = x.numpy()
        num_node = features.shape[0]
        node_type = "actor"

        with open(self.raw_file_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones(len(row))
        edge_type = "actor__to__actor"

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
            train_masks, val_masks, test_masks = [], [], []
            for f in self.raw_file_paths[2:]:
                tmp = np.load(f)
                train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
                val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
                test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
            train_mask = train_masks[self._split_id]
            val_mask = val_masks[self._split_id]
            test_mask = test_masks[self._split_id]

            train_idx = torch.nonzero(train_mask == 1).reshape(-1)
            val_idx = torch.nonzero(val_mask == 1).reshape(-1)
            test_idx = torch.nonzero(test_mask == 1).reshape(-1)

        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
