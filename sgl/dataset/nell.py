import networkx as nx
import os
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import shutil
import torch
from torch_geometric.data import extract_tar
from torch_geometric.io import read_txt_array

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


class Nell(NodeDataset):
    def __init__(self, name="nell.0.001", root='./', split="official"):
        if name not in ["nell.0.1", "nell.0.01", "nell.0.001"]:
            raise ValueError("Dataset name not supported!")
        super(Nell, self).__init__(root + "Nell/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(split)

    @property
    def raw_file_paths(self):
        filenames = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [osp.join(self._raw_dir, f'ind.{self._name}.{filename}') for filename in filenames]

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        url = 'http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz'
        path = osp.join(self._raw_dir, "nell_data.tar.gz")
        print(url)
        download_to(url, path)
        extract_tar(path, self._raw_dir)
        os.unlink(path)

        raw_dir_path = osp.join(self._raw_dir, "nell_data")
        for root, dirs, files in os.walk(raw_dir_path, topdown=False):
            for file in files:
                if self._name in file:
                    shutil.move(osp.join(root, file), self._raw_dir)
        shutil.rmtree(raw_dir_path)

    def _process(self):
        objects = []
        for raw_file in self.raw_file_paths[:-1]:
            objects.append(pkl_read_file(raw_file))

        for i in range(len(objects) - 1):
            tmp = objects[i]
            tmp = tmp.todense() if hasattr(tmp, 'todense') else tmp
            objects[i] = torch.Tensor(tmp)

        x, tx, allx, y, ty, ally, graph = tuple(objects)
        test_index = read_txt_array(osp.join(self._raw_dir, f"ind.{self._name}.test.index"), dtype=torch.long)

        sorted_test_index = test_index.sort()[0]

        tx_ext = torch.zeros(len(graph) - allx.shape[0], x.shape[1])
        tx_ext[sorted_test_index - allx.shape[0]] = tx

        ty_ext = torch.zeros(len(graph) - ally.shape[0], y.shape[1])
        ty_ext[sorted_test_index - ally.shape[0]] = ty

        tx, ty = tx_ext, ty_ext

        # PyG creates feature vectors for relations here. Whereas the augmented features consume
        # too much memory to be stored in a dense matrix (numpy matrix), thus we ignore that.
        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]
        x = x.numpy()

        y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
        y[test_index] = y[sorted_test_index]
        y = torch.LongTensor(y)

        adj = sp.coo_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)))
        row, col, edge_weight = adj.row, adj.col, adj.data

        num_node = y.shape[0]
        node_type = "knowledge"
        edge_type = "knowledge__to__knowledge"

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=x, y=y)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "official":
            y = pkl_read_file(osp.join(self._raw_dir, f"ind.{self._name}.y"))
            test_idx = read_txt_array(osp.join(self._raw_dir, f"ind.{self._name}.test.index"), dtype=torch.long)
            train_idx = torch.arange(y.shape[0], dtype=torch.long)
            val_idx = torch.arange(y.shape[0], y.shape[0] + 500, dtype=torch.long)

        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
