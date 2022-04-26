import numpy as np
import os.path as osp
import pickle as pkl
import torch
from torch_sparse import coalesce

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


class WebKB(NodeDataset):
    # Have 10 different split of training and validation set, identified by split_id in [0, 9]
    # Currently, we only support calculating the accuracy of one split, 
    # and average accuracy will be supported in the future.
    def __init__(self, name="cornell", root="./", split="official", split_id=0):
        name = name.lower()
        if name not in ['cornell', 'texas', 'wisconsin']:
            raise ValueError("Dataset name not supported!")
        if split_id not in range(10):
            raise ValueError("Split id not supported")

        super(WebKB, self).__init__(root + "WebKB", name)
        self._data = pkl_read_file(self.processed_file_paths)
        self._split, self._split_id = split, split_id
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        filenames = ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                     ] + [f'{self._name}_split_0.6_0.2_{i}.npz' for i in range(10)]
        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        return osp.join(self._processed_dir, f"{self._name}.graph")

    def _download(self):
        url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

        for f in self.raw_file_paths[:2]:
            raw_file_name = osp.basename(f)
            path = osp.join(self._raw_dir, raw_file_name)
            file_url = f'{url}/new_data/{self._name}/{raw_file_name}'
            print(file_url)
            download_to(file_url, path)

        for f in self.raw_file_paths[2:]:
            raw_file_name = osp.basename(f)
            path = osp.join(self._raw_dir, raw_file_name)
            file_url = f'{url}/splits/{raw_file_name}'
            print(file_url)
            download_to(file_url, path)

    def _process(self):
        with open(self.raw_file_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            features = np.array(x)
            num_node = features.shape[0]
            node_type = "page"

            y = [int(r.split('\t')[2]) for r in data]
            labels = torch.LongTensor(y)

        with open(self.raw_file_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, num_node, num_node)

        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones(len(row))
        edge_type = "page__to__page"

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
