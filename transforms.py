import copy
import torch
import numpy as np
from typing import Tuple, Dict, Optional
from torch import BoolTensor, Tensor
from sgl.data.base_data import Node, Edge, Graph

def random_drop_edges(eg: Edge, num_node: int, p: float = 0.5, force_undirected: bool = True) -> Edge:
    """
    e - source edges; p - dropping probability; force_undirected - drop edges symmetrically
    adapted from PyG's dropout_edge function
    """
    num_edge = eg.num_edge

    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1!')

    if p == 0.0:
        return eg

    edge_mask = torch.randn(num_edge) >= p

    return drop_edges(eg, num_node, edge_mask, force_undirected=force_undirected)

def biased_drop_edges(eg: Edge, num_node: int, edge_mask: BoolTensor) -> Edge:
    """
    If edge_mask[i] = False, then the i-th edge will be dropped.
    """
    num_edge = eg.num_edge

    if (edge_mask.dim() > 1) or (edge_mask.size(0) != num_edge):
        raise ValueError('the shape of edge drop mask is wrong!')

    return drop_edges(eg, num_node, edge_mask)

def random_drop_nodes(g: Graph, p: float = 0.5, keep_ids: bool = False) -> Tuple[Graph, BoolTensor]:
    """
    g - original graph; p - dropping probability 
    return new graph and node mask (node_mask[i] == True means the i-th node is preserved)
    """
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1!')

    num_node = g.num_node

    if p == 0.0:
        return g

    node_mask = torch.randn(num_node) >= p

    new_g = get_subgraph(g, node_mask)

    return new_g, node_mask

def drop_edges(old_eg: Edge, num_node: int, edge_mask: BoolTensor, \
                force_undirected: bool = False, node_id_dict=None) -> Edge:
    edge_index = old_eg.edge_index
    edge_weight = old_eg.edge_weight

    row, col = edge_index

    edge_index = torch.vstack([row, col])

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]
    edge_weight = edge_weight[edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight])

    row, col = edge_index

    if node_id_dict is not None:
        assert isinstance(node_id_dict, Dict), 'node_id_dict must be of Class Dict'

        for i in range(edge_index.shape[1]):
            row[i] = node_id_dict[row[i].item()]
            col[i] = node_id_dict[col[i].item()]

    return Edge(row, col, edge_weight, old_eg.edge_type, num_node, old_eg.edge_attrs)

def add_edges(old_eg: Edge, num_node: int, add_indices: Tensor, \
                add_weights: Optional[Tensor] = None, del_repeated: bool = False) -> Edge:
    """
    if del_repeated == True, then delete repeated edges
    if force_undirected == True, then make edges undirected
    if add_weights is not provided, then it will be ones
    """
    edge_index = old_eg.edge_index
    edge_weight = old_eg.edge_weight
    edge_type = old_eg.edge_type
    edge_attrs = old_eg.edge_attrs

    if add_indices.dim() != 2 or add_indices.shape[0] != 2:
        raise ValueError('add indices must be in shape of (2, x)!')

    row, col = edge_index
    added_row, added_col = add_indices
    if added_row.max().item() >= num_node or added_col.max().item() >= num_node:
        raise ValueError('indices must be in range of [0, num_node)!')
    if added_row.min().item() < 0 or added_col.min().item() < 0:
        raise ValueError('indices must be in range of [0, num_node)!')
    
    if add_weights is None:
        add_weights = torch.ones_like(added_row, dtype=torch.float)

    row, col = torch.cat([row, added_row]), torch.cat([col, added_col])
    edge_weight = torch.cat([edge_weight, add_weights])

    new_eg = Edge(row, col, edge_weight, edge_type, num_node, edge_attrs)

    if del_repeated:
        return delete_repeated_edges(new_eg, num_node)
    else:
        return new_eg

def delete_repeated_edges(old_eg: Edge, num_node: int) -> Edge:
    """
    delete repeated edges, such as (0, 2) and (0, 2)
    """
    sorted_eg = sort_edges(old_eg, num_node)
    
    row, col = sorted_eg.edge_index
    edge_weight = sorted_eg.edge_weight
    edge_type = sorted_eg.edge_type
    edge_attrs = sorted_eg.edge_attrs

    row = row.numpy()
    col = col.numpy()
    edge_weight = edge_weight.numpy()
    
    idx = row * num_node + col
    _, indcies = np.unique(idx, return_index=True)
    row, col = row[indcies], col[indcies]
    edge_weight = edge_weight[indcies]

    return Edge(row, col, edge_weight, edge_type, num_node, edge_attrs)

def mask_features(node: Node, feature_mask: BoolTensor, type: int = 0) -> Tensor:
    """
    mask node features by row / column / element
    type == 0: by row, if feature_mask[i] == True, the i-th row will be masked as zero
    type == 1: by column, if feature_mask[j] == True, the j-th column will be masked as zero
    type == 2: by element, if feature_mask[i][j] == True, the (i, j) element will be masked as zero
    """
    assert node.x is not None, 'make sure that node has feature matrix'
    x = node.x.clone()

    N, f = x.size(0), x.size(1)
    if type == 0:
        assert feature_mask.size(0) == N, 'dimension does not match'
        x[feature_mask, :] = 0
    elif type == 1:
        assert feature_mask.size(0) == f, 'dimension does not match'
        x[:, feature_mask] = 0
    elif type == 2:
        assert feature_mask.size(0) == N, 'dimension does not match'
        assert feature_mask.size(1) == f, 'dimension does not match'
        x[feature_mask] = 0
    else:
        raise ValueError('the choice of type is 0, 1, or 2!')

    return x
    
def get_subgraph(g: Graph, node_mask: BoolTensor, keep_ids: bool = False) -> Graph:
    """
    Preserve nodes in the remain_node_list and edges connecting them.
    If k in remain_node_list, then the k-th node will be preserved.
    """
    num_node = g.num_node
    node_type = g.node_type
    edge_type = g.edge_type
    edge_index = g.edge_index

    if isinstance(node_mask, list):
        node_mask = np.array(node_mask)
    if isinstance(node_mask, BoolTensor):
        node_mask = node_mask.numpy()

    if (node_mask.dim() > 1) or (node_mask.size(0) != num_node):
        raise ValueError('the shape of node mask is wrong!')
    
    num_remain_node = node_mask.nonzero(as_tuple=True)[0].size(0)

    if num_remain_node == 0:
        # return an empty graph
        return Graph([], [], [], 0, node_type, edge_type)

    x = g.x
    y = g.y

    if keep_ids:
        drop_mask = ~node_mask
        if x is not None:
            x[drop_mask, :] = 0

        row, col = edge_index
        edge_mask = node_mask[row] & node_mask[col]
        new_edge = drop_edges(g.edge, num_node, edge_mask)
    
        new_node = Node(node_type, num_node, x, y)

    else:
        if x is not None:
            x = x[node_mask, :]
        if y is not None:
            y = y[node_mask]

        row, col = edge_index
        remain_nodes = np.arange(num_node)[node_mask]
        node_id_dict = {}
        for i, idx in enumerate(remain_nodes):
            node_id_dict[idx] = i
        edge_mask = node_mask[row] & node_mask[col]
        new_edge = drop_edges(g.edge, num_remain_node, edge_mask, node_id_dict=node_id_dict)
    
        new_node = Node(node_type, num_remain_node, x, y)

    new_g = copy.deepcopy(g)
    new_g.node = new_node
    new_g.edge = new_edge

    return new_g

def sort_edges(old_eg: Edge, num_node: int, sort_by: bool = True) -> Edge:
    """
    sort edges by row (or col)
    sort_by == True: sort by row, otherwise by column
    """
    row, col, edge_weight = old_eg.row, old_eg.col, old_eg.edge_weight
    row = row.numpy()
    col = col.numpy()
    edge_weight = edge_weight.numpy()
    if sort_by:
        idx = row * num_node + col
    else:
        idx = col * num_node + row
    
    perm = idx.argsort()
    row, col, edge_weight = row[perm], col[perm], edge_weight[perm]
    edge_type = old_eg.edge_type
    edge_attrs = old_eg.edge_attrs

    new_eg = Edge(row, col, edge_weight, edge_type, num_node, edge_attrs)   

    return new_eg

def add_self_loops(old_eg: Edge, num_node: int, add_weights: Optional[Tensor] = None) -> Edge:
    """
    add self loop edges, e.g. (0, 0), (1, 1)
    """
    if (add_weights is not None) and (add_weights.size(0) != num_node):
        raise ValueError('add weights must be in shape of [num_node]!')
    
    row = torch.arange(num_node)
    add_indices = torch.vstack([row, row])

    return add_edges(old_eg, num_node, add_indices, add_weights)

def remove_self_loops(old_eg: Edge, num_node: int) -> Edge:
    """
    remove self loop edge, e.g. (0, 0), (1, 1)
    """
    num_edge = old_eg.num_edge
    row, col = old_eg.row, old_eg.col
    drop_mask = torch.ones((num_edge, ), dtype=bool)
    drop_mask[row == col] = False

    return drop_edges(old_eg, num_node, drop_mask)
    

    
