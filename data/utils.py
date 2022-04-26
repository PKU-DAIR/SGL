import os.path as osp
import torch


def file_exist(filepaths):
    if isinstance(filepaths, list):
        for filepath in filepaths:
            if not osp.exists(filepath):
                return False
        return True
    else:
        if osp.exists(filepaths):
            return True
        else:
            return False


def to_undirected(edge_index):
    row, col = edge_index
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)

    return new_edge_index
