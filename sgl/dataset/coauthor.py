import os.path as osp
import pickle as pkl
import torch

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to, read_npz, random_split_dataset


class Coauthor(NodeDataset):
    def __init__(self, name="cs", root="./", split="random"):
        if name not in ['cs', 'phy']:
            raise ValueError("Dataset name not supported!")
        super(Coauthor, self).__init__(root + "coauthor/", name)
        self._data = pkl_read_file(self.processed_file_paths)
        self._split = split
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(split)

    @property
    def raw_file_paths(self):
        return osp.join(self._raw_dir, f'ms_academic_{self.name.lower()}.npz')

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz"
        file_url = url + '/' + osp.basename(self.raw_file_paths)
        print(file_url)
        download_to(file_url, self.raw_file_paths)

    def _process(self):
        x, edge_index, y = read_npz(self.raw_file_paths)
        num_node = x.shape[0]
        row, col = edge_index
        edge_weight = torch.ones(size=(len(row),))
        node_type = "paper"
        edge_type = "paper__to__paper"
        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=x.numpy(), y=y)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        if split == "random":
            train_idx, val_idx, test_idx = random_split_dataset(self.num_node)
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
