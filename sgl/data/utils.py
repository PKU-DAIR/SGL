import torch
import numpy as np
import os.path as osp

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

class Loader:
    def __init__(self, seed_nodes, batch_size):
        self.seed_nodes = seed_nodes
        self.batch_size = batch_size

    def __iter__(self):
        pass

    def __call__(self):
        pass
         
class RandomLoader(Loader):
    def __init__(self, seed_nodes, batch_size):
        super().__init__(seed_nodes, batch_size)
        self.num_batches = (len(seed_nodes) + batch_size - 1) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = np.random.choice(
                self.seed_nodes, self.batch_size, replace=False)
            yield batch

    def __call__(self):
        batch = np.random.choice(
                self.seed_nodes, self.batch_size, replace=False)
        
        return np.sort(batch)
    
class SplitLoader(Loader):
    def __init__(self, seed_nodes, batch_size):
        super().__init__(seed_nodes, batch_size)
        if not isinstance(seed_nodes, torch.LongTensor):
            seed_nodes = torch.LongTensor(seed_nodes)
        self.batches = torch.split(seed_nodes, self.batch_size)

    def __iter__(self, *args, **kwargs):
        for batch in self.batches:
            yield batch.numpy()

    def __len__(self):
        return len(self.batches)

    def __call__(self, bid, *args, **kwargs):
        return self.batches[bid]