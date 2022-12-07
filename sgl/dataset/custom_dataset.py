import torch
import numpy as np
import pickle as pkl
import os.path as osp
import scipy.sparse as sp
from typing import Tuple, List

from sgl.data.base_data import Graph, HeteroGraph
from sgl.data.base_dataset import NodeDataset, HeteroNodeDataset
from sgl.dataset.utils import pkl_read_file, file_exist, load_np

class Custom_Homo(NodeDataset):
    def __init__(self, name, node_type, edge_type_tuple, num_node = 0, root="./", splitted=True):
        self._num_node = num_node
        self._node_type = node_type
        if len(edge_type_tuple) != 3:
            raise ValueError('number of elements is invalid for input tuple')
        self._edge_type_tuple = edge_type_tuple
        super(Custom_Homo, self).__init__(root, name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(splitted)

    @property
    def raw_file_paths(self):
         # the storage path of user-defined raw data is 'root/name/raw/'
        return self._raw_dir

    @property
    def processed_file_paths(self):
        # the storage path of user-defined processed data is 'root/name/processed/name.graph'
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        pass 

    def _process(self):
        features, row, col, edge_weight, labels = None, None, None, None, None
        if file_exist(osp.join(self.raw_file_paths, "x.npy")):
            features = load_np(osp.join(self.raw_file_paths, "x.npy"))
        if features is not None:
            if self._num_node:
                assert self._num_node == features.shape[0], 'every node should have a feature vector'
            else:
                self._num_node = features.shape[0]
        elif not self._num_node:
            raise ValueError('please provide either feature matrix or number of node')

        num_node = self._num_node
        node_type = self._node_type

        if file_exist(osp.join(self.raw_file_paths, "adj_matrix.npz")):
            f = load_np(osp.join(self.raw_file_paths, "adj_matrix.npz"))
            row, col, edge_weight = f['row'], f['col'], f['data']
        else:
            raise ValueError('the adjacency matrix in coo-format is necessary')

        edge_type = self._edge_type_tuple[0] + '__to__' + self._edge_type_tuple[2]

        if file_exist(osp.join(self.raw_file_paths, "label.npy")):
            labels = load_np(osp.join(self.raw_file_paths, "label.npy"))
            if labels.ndim == 2:
                labels = np.argmax(labels, 1)
            labels = torch.LongTensor(labels)

        g = Graph(row, col, edge_weight, num_node, node_type, edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)


    def __generate_split(self, splitted):
        train_idx, val_idx, test_idx = None, None, None
        if splitted and file_exist(osp.join(self.raw_file_paths, "indices.npz")):
            split_indices = load_np(osp.join(self.raw_file_paths, "indices.npz"))
            if 'train_idx' in split_indices:
                train_idx = split_indices['train_idx']
            if 'val_idx' in split_indices:
                val_idx = split_indices['val_idx']
            if 'test_idx' in split_indices:
                test_idx = split_indices['test_idx']
        
        return train_idx, val_idx, test_idx


class Custom_Hetero(HeteroNodeDataset):
    EDGE_TYPE_DELIMITER = '__to__'

    def __init__(self, name, type_of_node_to_predict: str, node_types: List[str], edge_types_tuple: List[Tuple], root="./", splitted=True):
        assert type_of_node_to_predict in node_types, 'make sure that the type of center node is in type list'
        self.NODE_TYPES = node_types
        self.TYPE_OF_NODE_TO_PREDICT = type_of_node_to_predict
        self.EDGE_TYPES_TUPLE = edge_types_tuple

        super(Custom_Hetero, self).__init__(root, name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(splitted)

    def edge_type_tuple_to_str(self, edge_type_tuple: Tuple) -> str:
        if len(edge_type_tuple) != 3:
            raise ValueError('number of elements is invalid for input tuple')
        return edge_type_tuple[0] + self.EDGE_TYPE_DELIMITER + edge_type_tuple[2]

    @property
    def raw_file_paths(self):
         # the storage path of user-defined raw data is 'root/name/raw/'
        return self._raw_dir

    @property
    def processed_file_paths(self):
        # the storage path of user-defined processed data is 'root/name/processed/name.graph'
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        pass 

    def _process(self):
        x, edge_indices, y, num_node_dict = {}, {}, {}, {}
        
        # obtain num_node_dict

        if file_exist(osp.join(self.raw_file_paths, "num_node.npz")):
            temp_num_node_dict = load_np(osp.join(self.raw_file_paths, "num_node.npz"))
            assert len(temp_num_node_dict) == len(self.NODE_TYPES), 'the length of num_node dict and node types don\'t match'
        else:
            raise ValueError('please provide the number of each type of nodes')
        
        for node_type in self.NODE_TYPES:
            num_node_dict[node_type] = int(temp_num_node_dict[node_type])
            print(f'number of {node_type} node is {num_node_dict[node_type]}')

        # obtain row_dict, col_dict, edge_weight_dict
        previous_node_cnt_dict = {}
        previous_node_cnt = 0
        for node_type in self.NODE_TYPES:
            previous_node_cnt_dict[node_type] = previous_node_cnt
            previous_node_cnt += num_node_dict[node_type]

        row_dict = {}
        col_dict = {}
        edge_weight_dict = {}
        if file_exist(osp.join(self.raw_file_paths, "edge_indices.npz")):
            edge_indices = load_np(osp.join(self.raw_file_paths, "edge_indices.npz"))
        else:
            raise ValueError('please provide edge indicies of the graph')
        for edge_type_tuple in self.EDGE_TYPES_TUPLE:
            edge_type = self.edge_type_tuple_to_str(edge_type_tuple)
            row_dict[edge_type] = edge_indices[edge_type][0] \
                                  + previous_node_cnt_dict[edge_type_tuple[0]]

            col_dict[edge_type] = edge_indices[edge_type][1] \
                                  + previous_node_cnt_dict[edge_type_tuple[2]]
            edge_weight_dict[edge_type] = torch.ones(edge_indices[edge_type].shape[1])

        # obtain node_types
        node_types = self.NODE_TYPES

        # obtain edge_types_found
        edge_types_found = [self.edge_type_tuple_to_str(
            edge_type_tuple) for edge_type_tuple in self.EDGE_TYPES_TUPLE]

        # obtain node_id_dict
        node_id_dict = {}
        accumulated_node_cnt = 0
        for node_type in self.NODE_TYPES:
            num_cur_node_type = num_node_dict[node_type]
            node_id_dict[node_type] = [i for i in range(
                accumulated_node_cnt,
                accumulated_node_cnt + num_cur_node_type)]
            accumulated_node_cnt += num_cur_node_type

        # obtain x_dict
        x_dict = {}
        total_dim = 0
        start_pos_of_node_type = {}

        if file_exist(osp.join(self.raw_file_paths, "x.npz")):
            x = load_np(osp.join(self.raw_file_paths, "x.npz"))

        for node_type in self.NODE_TYPES:
            start_pos_of_node_type[node_type] = total_dim
            if len(x) > 0 and node_type in x:
                total_dim += x[node_type].shape[1]

        accumulated_feature_dim = 0
        for node_type in self.NODE_TYPES:
            padded_tensor = np.zeros((num_node_dict[node_type], total_dim))
            if len(x) > 0 and node_type in x:
                cur_x_len = x[node_type].shape[1]
                padded_tensor[:, accumulated_feature_dim:accumulated_feature_dim \
                                                         + cur_x_len] = x[node_type]
                accumulated_feature_dim += cur_x_len
            x_dict[node_type] = padded_tensor
        
        # obtain y_dict
        y_dict = {}

        if file_exist(osp.join(self.raw_file_paths, "label.npz")):
            y = load_np(osp.join(self.raw_file_paths, "label.npz"))
        
        for node_type in self.NODE_TYPES:
            if len(y) > 0 and node_type in y:
                y_dict[node_type] = torch.tensor(y[node_type])
            else:
                y_dict[node_type] = None

        g = HeteroGraph(row_dict, col_dict, edge_weight_dict, num_node_dict, node_types, edge_types_found, node_id_dict,
                        x_dict, y_dict)

        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, splitted):
        train_idx, val_idx, test_idx = None, None, None
        if splitted and file_exist(osp.join(self.raw_file_paths, "indices.npz")):
            split_indices = load_np(osp.join(self.raw_file_paths, "indices.npz"))
            if 'train_idx' in split_indices:
                train_idx = split_indices['train_idx']
            if 'val_idx' in split_indices:
                val_idx = split_indices['val_idx']
            if 'test_idx' in split_indices:
                test_idx = split_indices['test_idx']
        
        return train_idx, val_idx, test_idx

