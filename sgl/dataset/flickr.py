import json
import numpy as np
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import torch

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


class Flickr(NodeDataset):
    def __init__(self, name="flickr", root="./", split="official"):
        super(Flickr, self).__init__(root + "Flickr", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        filenames = ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']
        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        return osp.join(self._processed_dir, f"{self._name}.graph")

    def _download(self):
        url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

        adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
        feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
        class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
        role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'

        path = osp.join(self._raw_dir, 'adj_full.npz')
        file_url = url.format(adj_full_id)
        print(file_url)
        download_to(file_url, path)

        path = osp.join(self._raw_dir, 'feats.npy')
        file_url = url.format(feats_id)
        print(file_url)
        download_to(file_url, path)

        path = osp.join(self._raw_dir, 'class_map.json')
        file_url = url.format(class_map_id)
        print(file_url)
        download_to(file_url, path)

        path = osp.join(self._raw_dir, 'role.json')
        file_url = url.format(role_id)
        print(file_url)
        download_to(file_url, path)

    def _process(self):
        features = np.load(osp.join(self._raw_dir, 'feats.npy'))
        num_node = features.shape[0]
        node_type = "image"

        f = np.load(osp.join(self._raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()

        row, col, edge_weight = adj.row, adj.col, adj.data
        edge_type = "image__to__image"

        ys = [-1] * num_node
        with open(osp.join(self._raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        labels = torch.LongTensor(ys)

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
            with open(osp.join(self._raw_dir, 'role.json')) as f:
                role = json.load(f)

            train_idx, val_idx, test_idx = role['tr'], role['va'], role['te']
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
