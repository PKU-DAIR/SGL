import torch
from scipy.sparse import csr_matrix
from torch import Tensor
import numpy as np


# Base class for adjacency matrix
class Edge:
    def __init__(self, row, col, edge_weight, edge_type, edge_attrs=None):
        if not isinstance(edge_type, str):
            raise TypeError("Edge type must be a string!")
        self.__edge_type = edge_type

        if (not isinstance(row, (list, np.ndarray, Tensor))) or (not isinstance(col, (list, np.ndarray, Tensor))) or (
                not isinstance(edge_weight, (list, np.ndarray, Tensor))):
            raise TypeError("Row, col and edge_weight must be a list, np.ndarray or Tensor!")
        self.__row = row
        self.__col = col
        self.__edge_weight = edge_weight
        self.__edge_attrs = edge_attrs
        self.__num_edge = len(row)

        if isinstance(row, Tensor) or isinstance(col, Tensor):
            self.__sparse_matrix = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())))
        else:
            self.__sparse_matrix = csr_matrix((edge_weight, (row, col)))

    @property
    def sparse_matrix(self):
        return self.__sparse_matrix

    @property
    def edge_type(self):
        return self.__edge_type

    @property
    def num_edge(self):
        return self.__num_edge

    @property
    def edge_index(self):
        return self.__row, self.__col

    @property
    def edge_attrs(self):
        return self.edge_attrs

    @edge_attrs.setter
    def edge_attrs(self, edge_attrs):
        # more restrictions

        self.__edge_attrs = edge_attrs

    @property
    def row(self):
        return self.__row

    @property
    def col(self):
        return self.__col

    @property
    def edge_weight(self):
        return self.__edge_weight


# Base class or storing node information
class Node:
    def __init__(self, node_type, num_node, x=None, y=None, node_ids=None):
        if not isinstance(num_node, int):
            raise TypeError("Num nodes must be a integer!")
        elif not isinstance(node_type, str):
            raise TypeError("Node type must be a string!")
        elif (node_ids is not None) and (not isinstance(node_ids, (list, np.ndarray, Tensor))):
            raise TypeError("Node IDs must be a list, np.ndarray or Tensor!")
        self.__num_node = num_node
        self.__node_type = node_type
        if node_ids is not None:
            self.__node_ids = node_ids
        else:
            self.__node_ids = range(num_node)
        self.__x = x
        self.__y = y

    @property
    def num_node(self):
        return self.__num_node

    @property
    def node_type(self):
        return self.__node_type

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        # more restrictions

        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        # more restrictions

        self.__y = y


# Base class for homogeneous graph
class Graph:
    def __init__(self, row, col, edge_weight, num_node, node_type, edge_type, x=None, y=None, node_ids=None,
                 edge_attr=None):
        self.__edge = Edge(row, col, edge_weight, edge_type, edge_attr)
        if node_ids is None:
            self.__node_ids = range(num_node)
        else:
            self.__node_ids = node_ids
        self.__node = Node(node_type, num_node, x, y, node_ids)

    @property
    def num_node(self):
        return self.__node.num_node

    @property
    def num_edge(self):
        return self.__edge.num_edge

    @property
    def adj(self):
        return self.__edge.sparse_matrix

    @property
    def edge_type(self):
        return self.__edge.edge_type

    @property
    def node_type(self):
        return self.__node.node_type

    @property
    def x(self):
        return self.__node.x

    @x.setter
    def x(self, x):
        self.__node.x = x

    @property
    def y(self):
        return self.__node.y

    @y.setter
    def y(self, y):
        self.__node.y = y

    @property
    def num_features(self):
        if self.__node.x is not None:
            return self.__node.x.shape[1]

    @property
    def num_classes(self):
        if self.__node.y is not None:
            return self.__node.y.max() + 1

    @property
    def node_degrees(self):
        row_sum = self.adj.sum(axis=1)
        return torch.LongTensor(row_sum).squeeze(1)


# Base class for heterogeneous graph
class HeteroGraph:
    def __init__(self, rows, cols, edge_weights, num_nodes, node_types, edge_types, node_ids, edge_attrs=None,
                 xs=None, ys=None):
        self.__edges_dict = {}
        self.__edge_types = edge_types
        for edge_type in edge_types:
            if not isinstance(edge_type, str):
                raise TypeError("Edge type must be a string!")
        if (not isinstance(rows, dict)) or (not isinstance(cols, dict)) or (not isinstance(edge_weights, dict)) or (
                edge_attrs is not None and not isinstance(edge_attrs, dict)):
            raise TypeError("Rows, cols, edge weights and edge attrs must be dicts!")
        elif not isinstance(edge_types, list):
            raise TypeError("Edge types must be a list!")
        elif not ((rows.keys() == cols.keys()) and (cols.keys() == edge_weights.keys()) and (
                list(edge_weights.keys()).sort() == edge_types.sort())):
            raise ValueError("The keys of the rows, cols, edge_weights and edge_types must be the same!")

        for edge_type in edge_types:
            self.__edges_dict[edge_type] = Edge(rows[edge_type], cols[edge_type], edge_weights[edge_type],
                                                edge_type, edge_attrs.get(edge_type, None))

        self.__nodes_dict = {}
        self.__node_types = node_types
        for node_type in node_types:
            if not isinstance(node_type, str):
                raise TypeError("Node type must be a string!")
        if not isinstance(num_nodes, list):
            raise TypeError("Num nodes must be a list!")
        elif not isinstance(node_types, list):
            raise TypeError("Node types must be a list!")
        elif list(num_nodes.keys()).sort() != node_types.sort():
            raise TypeError("The keys of num_nodes and node_types must be the same!")
        elif ((xs is not None) and (not isinstance(xs, dict))) or ((ys is not None) and (not isinstance(ys, dict))):
            raise TypeError("Xs and Ys must be a dict!")

        if node_ids is None:
            self.__node_ids = {}
            for node_type in node_types:
                self.__node_ids[node_type] = range(num_nodes[node_type])
        else:
            self.__node_ids = node_ids

        for node_type in node_types:
            self.__nodes_dict[node_type] = Node(node_type, num_nodes[node_type], xs.get(node_type, None),
                                                ys.get(node_type, None), self.__node_ids[node_type])

    def __getitem__(self, key):
        if key in self.__edge_types:
            return self.__edges_dict[key]
        elif key in self.__node_types:
            return self.__nodes_dict[key]
        else:
            raise ValueError("Please input valid edge type or node type!")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Edge type or node type must be a string!")
        if key in self.__edge_types:
            if not isinstance(value, Edge):
                raise TypeError("Please organize the dataset using the Edge class!")
            # more restrictions

            self.__edges_dict[key] = value
        elif key in self.__node_types:
            if not isinstance(value, Node):
                raise TypeError("Please organize the dataset using the Node class!")
            # more restrictions

            self.__nodes_dict[key] = value
        else:
            raise ValueError("Please input valid edge type or node type!")

    @property
    def num_features(self):
        num_features = {}
        for node_type in self.__node_types:
            x_temp = self.__nodes_dict[node_type].x
            if x_temp is not None:
                num_features[node_type] = x_temp.shape[1]
            else:
                num_features[node_type] = 0
        return num_features

    @property
    def num_classes(self):
        num_classes = 0
        for node_type in self.__node_types:
            y_temp = self.__nodes_dict[node_type].y
            if y_temp is not None:
                num_classes = max(y_temp.max() + 1, num_classes)
        return num_classes

    @property
    def num_node(self):
        num_node = {}
        for node_type in self.__node_types:
            num_node[node_type] = self.__nodes_dict[node_type].num_node
        return num_node
