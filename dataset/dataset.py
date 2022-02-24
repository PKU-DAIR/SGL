import os
import os.path as osp

from utils import file_exist


# Base class for node-level tasks
class NodeDataset:
    def __init__(self, root, name):
        self._name = name
        self._root = osp.join(root, name)
        self._raw_dir = osp.join(self._root, "raw")
        self._processed_dir = osp.join(self._root, "processed")
        self._data = None
        self._train_idx, self._val_idx, self._test_idx = None, None, None
        self._preprocess()

    @property
    def name(self):
        return self._name

    @property
    def raw_file_paths(self):
        raise NotImplementedError

    @property
    def processed_file_paths(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError

    def _process(self):
        raise NotImplementedError

    def _preprocess(self):
        if file_exist(self.raw_file_paths):
            print(self._name)
            print("Files already downloaded.")
        else:
            print("Downloading...")
            if not file_exist(self._raw_dir):
                os.makedirs(self._raw_dir)
            self._download()
            print("Downloading done!")

        if file_exist(self.processed_file_paths):
            print("Files already processed.")
        else:
            print("Processing...")
            if not file_exist(self._processed_dir):
                os.makedirs(self._processed_dir)
            self._process()
            print("Processing done!")

    @property
    def data(self):
        return self._data

    @property
    def x(self):
        return self._data.x

    @x.setter
    def x(self, x):
        self._data.x = x

    @property
    def y(self):
        return self._data.y

    @y.setter
    def y(self, y):
        self._data.y = y

    @property
    def adj(self):
        return self._data.adj

    @property
    def edge_type(self):
        return self._data.edge_type

    @property
    def node_type(self):
        return self._data.node_type

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def test_idx(self):
        return self._test_idx

    @property
    def num_features(self):
        return self._data.num_features

    @property
    def num_classes(self):
        return self._data.num_classes

    @property
    def num_node(self):
        return self._data.num_node


# Base class for graph-level tasks
class GraphDataset:
    pass
