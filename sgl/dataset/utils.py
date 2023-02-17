import numpy as np
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import ssl
import sys
import torch
import urllib


def to_undirected(edge_index):
    row, col = edge_index
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)

    return new_edge_index


def remove_self_loops(edge_index):
    mask = (edge_index[0] != edge_index[1])
    new_edge_index = edge_index[:, mask]
    return new_edge_index


def download_to(url, path):
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as wf:
        try:
            wf.write(data.read())
        except IOError as e:
            print(e)
            exit(1)


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


def pkl_read_file(filepath):
    file = None
    with open(filepath, 'rb') as rf:
        try:
            if sys.version_info > (3, 0):
                file = pkl.load(rf, encoding="latin1")
            else:
                file = pkl.load(rf)
        except IOError as e:
            print(e)
            exit(1)
    return file

def load_np(path):
    f = np.load(path)
    return f

def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']), f['attr_shape']).todense()
    x = torch.from_numpy(x).to(torch.float)
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']), f['adj_shape']).tocoo()
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    y = torch.from_numpy(f['labels']).to(torch.long)

    return x, edge_index, y


def random_split_dataset(n_samples):
    val_idx = np.random.choice(list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    remain = set(range(n_samples)) - set(val_idx)
    test_idx = np.random.choice(list(remain), size=int(n_samples * 0.2), replace=False)
    train_idx = list(remain - set(test_idx))

    return train_idx, val_idx, test_idx
