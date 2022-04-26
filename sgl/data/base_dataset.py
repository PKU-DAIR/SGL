import itertools
import numpy as np
import os
import os.path as osp
import torch
import warnings
from scipy.sparse import csr_matrix

from sgl.data.base_data import Node, Edge
from sgl.data.utils import file_exist, to_undirected
from sgl.dataset.choose_edge_type import ChooseMultiSubgraphs


# Base class for node-level tasks
class NodeDataset:
    def __init__(self, root, name):
        self._name = name
        self._root = osp.join(root, name)
        self._raw_dir = osp.join(self._root, "raw")
        self._processed_dir = osp.join(self._root, "processed")
        self._data = None
        self._train_idx, self._val_idx, self._test_idx = None, None, None
        self.__preprocess()

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

    def __preprocess(self):
        if file_exist(self.raw_file_paths):
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
    def edge_type_cnt(self):
        return len(self.edge_types)

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


# Base class for heterogeneous node-level tasks
class HeteroNodeDataset:
    def __init__(self, root, name):
        self._name = name
        self._root = osp.join(root, name)
        self._raw_dir = osp.join(self._root, "raw")
        self._processed_dir = osp.join(self._root, "processed")
        self._data = None
        self._train_idx, self._val_idx, self._test_idx = None, None, None
        self.__preprocess()

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

    def __preprocess(self):
        if file_exist(self.raw_file_paths):
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

    def __getitem__(self, key):
        if key in self.data.edge_types:
            return self.data[key]
        elif key in self.data.node_types:
            return self.data[key]
        else:
            raise ValueError("Please input valid edge type or node type!")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Edge type or node type must be a string!")
        if key in self.data.edge_types:
            if not isinstance(value, Edge):
                raise TypeError(
                    "Please organize the dataset using the Edge class!")
            # more restrictions

            self.data.edges_dict[key] = value
        elif key in self.data.node_types:
            if not isinstance(value, Node):
                raise TypeError(
                    "Please organize the dataset using the Node class!")
            # more restrictions

            self.data.nodes_dict[key] = value
        else:
            raise ValueError("Please input valid edge type or node type!")

    @property
    def data(self):
        return self._data

    @property
    def y_dict(self):
        return self._data.y_dict

    @property
    def node_types(self):
        return self._data.node_types

    @property
    def edge_types(self):
        return self._data.edge_types

    @property
    def edge_type_cnt(self):
        return len(self.edge_types)

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def test_idx(self):
        return self._test_idx

    # return a sampled adjacency matrix containing edges of given edge types
    def sample_by_edge_type(self, edge_types, undirected=True):
        if not isinstance(edge_types, (str, list, tuple)):
            raise TypeError(
                "The given edge types must be a string or a list or a tuple!")
        elif isinstance(edge_types, str):
            edge_types = [edge_types]
        elif isinstance(edge_types, (list, tuple)):
            for edge_type in edge_types:
                if not isinstance(edge_type, str):
                    raise TypeError("Edge type must be a string!")

        pre_sampled_node_types = []
        for edge_type in edge_types:
            pre_sampled_node_types = pre_sampled_node_types + \
                                     [edge_type.split('__')[0], edge_type.split('__')[2]]
        pre_sampled_node_types = list(set(pre_sampled_node_types))

        sampled_node_types = []
        node_id_offsets = {}
        node_count = 0
        for node_type in self.node_types:
            if node_type in pre_sampled_node_types:
                sampled_node_types.append(node_type)

            node_id_offsets[node_type] = node_count
            node_count = node_count + self._data.num_node[node_type]

        num_node = 0
        feature = None
        node_id = None
        node_id_offset = {}
        for node_type in sampled_node_types:
            node_id_offset[node_type] = node_id_offsets[node_type] - num_node
            num_node = num_node + self._data.num_node[node_type]

            current_feature = torch.from_numpy(self._data[node_type].x)
            if current_feature is None:
                warnings.warn(
                    f'{node_type} nodes have no features!', UserWarning)
            if feature is None:
                feature = current_feature
            else:
                feature = torch.vstack((feature, current_feature))

            if node_id is None:
                node_id = self._data.node_id_dict[node_type][:]
            else:
                node_id = node_id + self._data.node_id_dict[node_type]

        rows, cols = None, None
        for edge_type in edge_types:
            row_temp, col_temp = self._data[edge_type].edge_index

            node_type_of_row = edge_type.split('__')[0]
            node_type_of_col = edge_type.split('__')[2]
            row_temp = row_temp - node_id_offset[node_type_of_row]
            col_temp = col_temp - node_id_offset[node_type_of_col]
            if undirected is True and node_type_of_row != node_type_of_col:
                row_temp, col_temp = to_undirected((row_temp, col_temp))

            if rows is None:
                rows, cols = row_temp, col_temp
            else:
                rows = torch.hstack((rows, row_temp))
                cols = torch.hstack((cols, col_temp))

        edge_weight = torch.ones(len(rows))
        adj = csr_matrix((edge_weight.numpy(), (rows.numpy(),
                                                cols.numpy())), shape=(num_node, num_node))

        # remove previously existed undirected edges
        adj.data = torch.ones(len(adj.data)).numpy()

        return adj, feature.numpy(), torch.LongTensor(node_id)

    # return sampled adjacency matrix containing the given meta-path, "xxx__to__xxx__to...__xxx"
    def sample_by_meta_path(self, meta_path, undirected=True):
        if isinstance(meta_path, str):
            if len(meta_path.split("__")) == 3:
                return self.sample_by_edge_type(meta_path, undirected)

        node_types = meta_path.split("__")

        node_type_st, node_type_ed = node_types[0], node_types[-1]
        sampled_node_types = []
        num_node = 0
        for node_type in self.node_types:
            if node_type in [node_type_st, node_type_ed]:
                sampled_node_types.append(node_type)
                num_node = num_node + self._data.num_node[node_type]

        feature = None
        node_id = None
        for node_type in sampled_node_types:
            current_feature = torch.from_numpy(self._data[node_type].x)
            if current_feature is None:
                warnings.warn(
                    f'{node_type} nodes have no features!', UserWarning)
            if feature is None:
                feature = current_feature
            else:
                feature = torch.vstack((feature, current_feature))

            if node_id is None:
                node_id = self._data.node_id_dict[node_type][:]
            else:
                node_id = node_id + self._data.node_id_dict[node_type]

        # two at a time
        adj = None
        for i in range(int((len(node_types) - 1) / 2)):
            edge_type = "__".join(
                [node_types[i * 2], "to", node_types[(i + 1) * 2]])
            row, col = self._data[edge_type].edge_index
            edge_weight = torch.ones(len(row))
            adj_temp = csr_matrix(
                (edge_weight.numpy(), (row.numpy(), col.numpy())))

            # extremely slow
            if adj is None:
                adj = adj_temp
            else:
                adj = adj * adj_temp

        adj = adj.tocoo()
        row, col, data = torch.LongTensor(adj.row), torch.LongTensor(
            adj.col), torch.FloatTensor(adj.data)

        st_index, ed_index = self.node_types.index(
            node_type_st), self.node_types.index(node_type_ed)
        if st_index == ed_index:
            for node_type in self.node_types[:st_index]:
                row = row - self._data.num_node[node_type]
                col = col - self._data.num_node[node_type]
        else:
            if st_index < ed_index:
                for node_type in self.node_types[:st_index]:
                    row = row - self._data.num_node[node_type]
                    col = col - self._data.num_node[node_type]
                for node_type in self.node_types[st_index:ed_index]:
                    col = col - self._data.num_node[node_type]
                col = col + self._data.num_node[node_type_st]
            else:
                for node_type in self.node_types[:ed_index]:
                    print(node_type)
                    col = col - self._data.num_node[node_type]
                    row = row - self._data.num_node[node_type]
                for node_type in self.node_types[ed_index:st_index]:
                    row = row - self._data.num_node[node_type]
                row = row + self._data.num_node[node_type_ed]

        if undirected is True:
            data = torch.ones(2 * len(data))
            row, col = to_undirected((row, col))
        adj = csr_matrix(
            (data.numpy(), (row.numpy(), col.numpy())), shape=(num_node, num_node))

        # remove existed self loops
        adj.data = torch.ones(len(adj.data)).numpy()
        return adj, feature.numpy(), torch.LongTensor(node_id)

    # return a dict of sub-graphs that contain all the combinations of given edge types and sampled number
    def nars_preprocess(self, edge_types, predict_class, random_subgraph_num, subgraph_edge_type_num):
        if not isinstance(edge_types, (str, list, tuple)):
            raise TypeError(
                "The given edge types must be a string or a list or a tuple!")
        elif isinstance(edge_types, str):
            edge_types = [edge_types]
        elif isinstance(edge_types, (list, tuple)):
            for edge_type in edge_types:
                if not isinstance(edge_type, str):
                    raise TypeError("Edge type must be a string!")

        adopted_edge_type_combinations = ChooseMultiSubgraphs(
            subgraph_num=random_subgraph_num,
            edge_type_num=subgraph_edge_type_num,
            edge_types=edge_types,
            predict_class=predict_class
        )

        if random_subgraph_num > len(adopted_edge_type_combinations):
            random_subgraph_num = len(adopted_edge_type_combinations)
            warnings.warn(
                "The input random_subgraph_num exceeds the number of all the combinations of edge types!"
                f"\nThe random_subgraph_num has been set to {len(adopted_edge_type_combinations)}.", UserWarning)

        chosen_idx = np.random.choice(np.arange(len(adopted_edge_type_combinations)), size=random_subgraph_num,
                                      replace=False)
        chosen_edge_types = [tuple(edge_type) for edge_type in np.array(
            adopted_edge_type_combinations)[chosen_idx]]
        subgraph_dict = {}
        for chosen_edge_type in chosen_edge_types:
            print(chosen_edge_type)
            subgraph_dict[chosen_edge_type] = self.sample_by_edge_type(
                chosen_edge_type)

        return subgraph_dict
