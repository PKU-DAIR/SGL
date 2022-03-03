import sys
import os.path as osp
import pickle as pkl
import ssl
import urllib
import torch


def to_undirected(edge_index):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index


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
            if osp.exists(filepath):
                return True
    else:
        if osp.exists(filepaths):
            return True

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
