import numpy as np
import os.path as osp
import pickle as pkl
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from sgl.data.base_data import Graph
from sgl.data.base_dataset import NodeDataset
from sgl.dataset.utils import pkl_read_file, download_to


# A variety of non-homophilous graph datasets
# Note that only "penn94" has official split_mask, split_id in [0, 9] to identify different 
# For other datasets applied official split, you should set numbers of samples for training/valid set
class LINKXDataset(NodeDataset):
    def __init__(self, name="penn94", root="./", split="official", split_id=0, num_train_per_class=10,
                 num_valid_per_class=10):
        name = name.lower()
        if name not in ['penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55']:
            raise ValueError("Dataset name not supported!")
        if name in ['penn94'] and split_id not in range(5):
            raise ValueError("Split id not supported!")
        super(LINKXDataset, self).__init__(root + 'LINKXDataset', name)

        self._data = pkl_read_file(self.processed_file_paths)
        self._split, self._split_id = split, split_id
        self._num_train_per_class = num_train_per_class
        self._num_valid_per_class = num_valid_per_class
        self._train_idx, self._val_idx, self._test_idx = self.__generate_split(
            split)

    @property
    def raw_file_paths(self):
        dataset_name = {
            'penn94': 'Penn94.mat',
            'reed98': 'Reed98.mat',
            'amherst41': 'Amherst41.mat',
            'cornell5': 'Cornell5.mat',
            'johnshopkins55': 'Johns%20Hopkins55.mat',
        }
        splits_name = {
            'penn94': 'fb100-Penn94-splits.npy',
        }
        filenames = [dataset_name[self._name]]
        if self._name in splits_name:
            filenames += [splits_name[self._name]]

        return [osp.join(self._raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        return osp.join(self._processed_dir, f"{self._name}.graph")

    def _download(self):
        url = 'https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data'

        datasets = {
            'penn94': f'{url}/facebook100/Penn94.mat',
            'reed98': f'{url}/facebook100/Reed98.mat',
            'amherst41': f'{url}/facebook100/Amherst41.mat',
            'cornell5': f'{url}/facebook100/Cornell5.mat',
            'johnshopkins55': f'{url}/facebook100/Johns%20Hopkins55.mat',
        }

        splits = {
            'penn94': f'{url}/splits/fb100-Penn94-splits.npy',
        }

        file_url = datasets[self._name]
        print(file_url)
        download_to(file_url, self.raw_file_paths[0])

        if self._name in splits:
            file_url = splits[self._name]
            print(file_url)
            download_to(file_url, self.raw_file_paths[1])

    def _process(self):
        mat = loadmat(self.raw_file_paths[0])

        edge_index = mat['A'].tocsr().tocoo()
        row, col = edge_index.row, edge_index.col
        edge_weight = torch.ones(len(row))
        edge_type = "user__to__user"

        metadata = torch.from_numpy(mat['local_info'].astype('int64'))

        xs = []
        labels = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
        for i in range(x.size(1)):
            _, out = x[:, i].unique(return_inverse=True)
            xs.append(F.one_hot(out).to(torch.float))
        x = torch.cat(xs, dim=-1)

        features = x.numpy()
        num_node = features.shape[0]
        node_type = "user"

        g = Graph(row, col, edge_weight, num_node,
                  node_type, edge_type, x=features, y=labels)

        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def __generate_split(self, split):
        have_split = ['penn94']
        # datasets that have official split_mask
        if self._name in have_split:
            if split == 'official':
                split_full = np.load(self.raw_file_paths[1], allow_pickle=True)

                split_idx = split_full[self._split_id]
                train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

            else:
                raise ValueError("Please input valid split pattern!")

        else:
            if split == 'official':

                mat = loadmat(self.raw_file_paths[0])
                metadata = mat['local_info'].astype('int64')
                labels = metadata[:, 1] - 1  # gender label, -1 means unlabeled

                num_train_per_class = self._num_train_per_class
                num_val = self._num_valid_per_class
                train_idx, val_idx, test_idx = np.empty(0), np.empty(0), np.empty(0)
                for i in range(self.num_classes):
                    idx = np.nonzero(labels == i)[0]
                    train_idx = np.append(train_idx, idx[:num_train_per_class])
                    val_idx = np.append(val_idx, idx[num_train_per_class: num_train_per_class + num_val])
                    test_idx = np.append(test_idx, idx[num_train_per_class + num_val:])
                train_idx.reshape(-1)
                val_idx.reshape(-1)
                test_idx.reshape(-1)

            else:
                raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
