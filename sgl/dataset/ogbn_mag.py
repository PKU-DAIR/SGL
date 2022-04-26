import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import torch
from ogb.nodeproppred import PygNodePropPredDataset

from sgl.data.base_data import HeteroGraph
from sgl.data.base_dataset import HeteroNodeDataset
from sgl.dataset.utils import pkl_read_file, to_undirected


class OgbnMag(HeteroNodeDataset):
    TYPE_OF_NODE_TO_PREDICT = 'paper'

    def __init__(self, name="mag", root=".", split="official"):
        if name not in ["mag"]:
            raise ValueError("Dataset name not found!")
        super(OgbnMag, self).__init__(root + "ogbn/", name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(split)

    @property
    def raw_file_paths(self):
        filepath = "ogbn_" + self._name + "/raw/geometric_data_processed.pt"
        return osp.join(self._raw_dir, filepath)

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self._processed_dir, "{}.{}".format(self._name, filename))

    @property
    def EDGE_TYPES(self):
        return self._data.edge_types

    def _download(self):
        dataset = PygNodePropPredDataset("ogbn-" + self._name, self._raw_dir)

    def _process(self):
        dataset = PygNodePropPredDataset("ogbn-" + self._name, self._raw_dir)

        data = dataset[0]
        node_types = list(data.num_nodes_dict.keys())
        edge_types = list(data.edge_index_dict.keys())

        # generate feature for nodes without features via averaging
        data.x_dict['paper'] = data.x_dict['paper'].numpy()

        row_author_temp = data.edge_index_dict[('author', 'writes', 'paper')][0, :].numpy()
        col_author_temp = data.edge_index_dict[('author', 'writes', 'paper')][1, :].numpy()
        edge_weight_author_temp = torch.ones(len(row_author_temp)).numpy()
        adj_author_temp = sp.csr_matrix((edge_weight_author_temp, (row_author_temp, col_author_temp)),
                                        shape=(data.num_nodes_dict['author'], data.num_nodes_dict['paper']))
        data.x_dict['author'] = adj_author_temp.dot(data.x_dict['paper'])

        row_inst_temp = data.edge_index_dict[('author', 'affiliated_with', 'institution')][1, :].numpy()
        col_inst_temp = data.edge_index_dict[('author', 'affiliated_with', 'institution')][0, :].numpy()
        edge_weight_inst_temp = torch.ones(len(row_inst_temp)).numpy()
        adj_inst_temp = sp.csr_matrix((edge_weight_inst_temp, (row_inst_temp, col_inst_temp)),
                                      shape=(data.num_nodes_dict['institution'], data.num_nodes_dict['author']))
        data.x_dict['institution'] = adj_inst_temp.dot(data.x_dict['author'])

        row_field_temp = data.edge_index_dict[('paper', 'has_topic', 'field_of_study')][1, :].numpy()
        col_field_temp = data.edge_index_dict[('paper', 'has_topic', 'field_of_study')][0, :].numpy()
        edge_weight_field_temp = torch.ones(len(row_field_temp)).numpy()
        adj_field_temp = sp.csr_matrix((edge_weight_field_temp, (row_field_temp, col_field_temp)),
                                       shape=(data.num_nodes_dict['field_of_study'], data.num_nodes_dict['paper']))
        data.x_dict['field_of_study'] = adj_field_temp.dot(data.x_dict['paper'])

        current_nodes = [0]
        current_nodes_dict = {}

        # the order of node types is important, which decides the node id
        num_node_dict, node_id_dict, x_dict, y_dict = {}, {}, {}, {}
        for i, node_type in enumerate(node_types):
            num_nodes_temp = data.num_nodes_dict[node_type]
            x_temp = data.x_dict.get(node_type, None)
            y_temp = data.y_dict.get(node_type, None)

            node_ids_temp = list(range(current_nodes[i], current_nodes[i] + data.num_nodes_dict[node_type]))
            current_nodes.append(current_nodes[-1] + data.num_nodes_dict[node_type])
            current_nodes_dict[node_type] = current_nodes[i]

            num_node_dict[node_type] = num_nodes_temp
            node_id_dict[node_type] = node_ids_temp
            x_dict[node_type] = x_temp
            y_dict[node_type] = y_temp

        edge_types_found = []
        row_dict, col_dict, edge_weight_dict = {}, {}, {}
        for i, edge_type in enumerate(edge_types):
            row_type = edge_type[0]
            col_type = edge_type[2]

            edge_type_used = "__".join([row_type, "to", col_type])
            edge_types_found.append(edge_type_used)

            row_temp = data.edge_index_dict[edge_type][0, :] + current_nodes_dict[row_type]
            col_temp = data.edge_index_dict[edge_type][1, :] + current_nodes_dict[col_type]

            if row_type != col_type:
                edge_weight_temp = torch.ones(len(row_temp))

                row_dict[edge_type_used] = row_temp
                col_dict[edge_type_used] = col_temp
                edge_weight_dict[edge_type_used] = edge_weight_temp

                reverse_edge_type_used = "__".join([col_type, "to", row_type])
                edge_types_found.append(reverse_edge_type_used)

                row_dict[reverse_edge_type_used] = col_temp
                col_dict[reverse_edge_type_used] = row_temp
                edge_weight_dict[reverse_edge_type_used] = edge_weight_temp

            else:
                row_temp, col_temp = to_undirected((row_temp, col_temp))
                edge_weight_temp = torch.ones(len(row_temp))

                row_dict[edge_type_used] = row_temp
                col_dict[edge_type_used] = col_temp
                edge_weight_dict[edge_type_used] = edge_weight_temp

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
            dataset = PygNodePropPredDataset("ogbn-" + self._name, self._raw_dir)
            split_idx = dataset.get_idx_split()
            train_idx, val_idx, test_idx = split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test'][
                'paper']
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
