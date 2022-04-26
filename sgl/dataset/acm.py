import os.path as osp
import pickle as pkl
import torch
from torch_geometric.datasets import HGBDataset
from typing import Tuple

from sgl.data.base_data import HeteroGraph
from sgl.data.base_dataset import HeteroNodeDataset
from sgl.dataset.utils import pkl_read_file


class Acm(HeteroNodeDataset):
    NODE_TYPES = [
        'paper',
        'author',
        'subject',
        'term'
    ]

    TYPE_OF_NODE_TO_PREDICT = 'paper'

    EDGE_TYPE_DELIMITER = '__to__'

    EDGE_TYPES_TUPLE = [
        ('paper', 'cite', 'paper'),
        ('paper', 'ref', 'paper'),
        ('paper', 'to', 'author'),
        ('author', 'to', 'paper'),
        ('paper', 'to', 'subject'),
        ('subject', 'to', 'paper'),
        ('paper', 'to', 'term'),
        ('term', 'to', 'paper')
    ]

    def __init__(self, root="./", split="official"):
        name = 'acm'
        # This dataset is processed based on the output of HGBDataset
        self.src_dataset = HGBDataset(root=root, name='acm').data

        super(Acm, self).__init__(root + "hgb/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    def edge_type_tuple_to_str(self, edge_type_tuple: Tuple) -> str:
        if len(edge_type_tuple) != 3:
            raise ValueError('number of elements is invalid for input tuple')
        return edge_type_tuple[0] + self.EDGE_TYPE_DELIMITER + edge_type_tuple[2]

    @property
    def raw_file_paths(self):
        filepath = "hgb_" + self._name + "/raw/geometric_data_processed.pt"
        return osp.join(self._raw_dir, filepath)

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    def _download(self):
        pass

    def _process(self):

        # obtain num_node_dict
        num_node_dict = {}
        for node_type in self.NODE_TYPES:
            if 'x' in self.src_dataset[node_type]:
                num_node_dict[node_type] = self.src_dataset[node_type]['x'].size(
                    0)
            else:
                num_node_dict[node_type] = self.src_dataset[node_type]['num_nodes']
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
        for edge_type_tuple in self.EDGE_TYPES_TUPLE:
            edge_type = self.edge_type_tuple_to_str(edge_type_tuple)

            row_dict[edge_type] = self.src_dataset[edge_type_tuple]['edge_index'][0] \
                                  + previous_node_cnt_dict[edge_type_tuple[0]]

            col_dict[edge_type] = self.src_dataset[edge_type_tuple]['edge_index'][1] \
                                  + previous_node_cnt_dict[edge_type_tuple[2]]

            edge_weight_dict[edge_type] = torch.ones(
                self.src_dataset[edge_type_tuple]['edge_index'].size(1))

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
        for node_type in self.NODE_TYPES:
            start_pos_of_node_type[node_type] = total_dim
            if 'x' in self.src_dataset[node_type]:
                total_dim += self.src_dataset[node_type]['x'].size(1)

        accumulated_feature_dim = 0
        for node_type in self.NODE_TYPES:
            padded_tensor = torch.zeros(num_node_dict[node_type], total_dim)
            if 'x' in self.src_dataset[node_type]:
                cur_x_len = self.src_dataset[node_type]['x'].size(1)
                padded_tensor[:, accumulated_feature_dim:accumulated_feature_dim +
                                                         cur_x_len] = self.src_dataset[node_type]['x']
                accumulated_feature_dim += cur_x_len
            x_dict[node_type] = padded_tensor.numpy()

        # obtain y_dict
        y_dict = {}

        for node_type in self.NODE_TYPES:
            if 'y' in self.src_dataset[node_type]:
                y_dict[node_type] = self.src_dataset[node_type]['y']
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

    def __generate_split(self, split):
        if split == "official":
            train_idx = self.src_dataset[self.TYPE_OF_NODE_TO_PREDICT]['train_mask'].nonzero(
            ).flatten()
            test_idx = self.src_dataset[self.TYPE_OF_NODE_TO_PREDICT]['test_mask'].nonzero(
            ).flatten()
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, None, test_idx
