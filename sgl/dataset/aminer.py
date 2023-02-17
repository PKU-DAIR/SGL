import os.path as osp
import pickle as pkl
import torch
from torch_geometric.datasets import AMiner
from typing import Tuple

from sgl.data.base_data import HeteroGraph
from sgl.data.base_dataset import HeteroNodeDataset
from sgl.dataset.utils import pkl_read_file

class Aminer(HeteroNodeDataset):
    NODE_TYPES = [
        'paper',
        'author',
        'venue'
    ]

    TYPE_OF_NODE_TO_PREDICT = ['author', 'venue']

    EDGE_TYPE_DELIMITER = '__to__'

    EDGE_TYPES_TUPLE = [
        ('paper', 'written_by', 'author'),
        ('author', 'writes', 'paper'),
        ('paper', 'published_in', 'venue'),
        ('venue', 'publishes', 'paper'),
    ]

    def __init__(self, root="./"):
        name = 'aminer'
        # This dataset is processed based on the output of AMinerDataset
        self.src_dataset = AMiner(root=root).data

        super(Aminer, self).__init__(root, name)

        self._data = pkl_read_file(self.processed_file_paths)  

    def edge_type_tuple_to_str(self, edge_type_tuple: Tuple) -> str:
        if len(edge_type_tuple) != 3:
            raise ValueError('number of elements is invalid for input tuple')
        return edge_type_tuple[0] + self.EDGE_TYPE_DELIMITER + edge_type_tuple[2]

    @property
    def raw_file_paths(self):
        filepath = self._root + "/processed/data.pt"
        return filepath

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

        for node_type in self.NODE_TYPES:
            x_dict[node_type] = None

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