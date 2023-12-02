from sgl.data.base_data import Block
import sgl.models.simple_models as SimpleModels
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp, RwGraphOp
from sgl.utils import sparse_mx_to_torch_sparse_tensor

import torch
import itertools
import numpy as np
import concurrent.futures

class LazyGNN(BaseSAMPLEModel):
    def __init__(self, dataset, training_sampler, eval_sampler=None, hidden_dim=128, basemodel="GCN", dropout=0.5, num_layers=2, max_workers=5, max_threads=-1, rho=1.1, tau=2, device="cpu"):
        super(LazyGNN, self).__init__()
        if basemodel == "SAGE":
            self._pre_graph_op = RwGraphOp()
        elif basemodel == "GCN":
            self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._max_workers = max_workers
        self._max_threads = max_threads if max_threads > -1 else torch.get_num_threads() // 2
        self._device = device
        # hyperparameters for recycling
        self._rho = rho
        self._tau = tau 
        # define the base model
        self._base_model = getattr(SimpleModels, basemodel)(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)

    def preprocess(self, adj, x, val_dataloader=None, test_dataloader=None):
        if val_dataloader is None:
            norm_adj = self._pre_graph_op._construct_adj(adj)
            norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
            # if evaluation on full-batch, then we can pre-move the full feature/adjacency matrix to the device to save time
            self._processed_block = Block(norm_adj)
            self._processed_block.to_device(self._device)
            self._processed_feature = x.to(self._device)
        else:
            # If val/test_dataloader is provided, it means that we conduct minibatch evaluation.
            # In such case, we could prepare evaluation minibatches in advance.
            self._val_samples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(torch.get_num_threads()*0.4)) as executor:
                self._val_sampling_jobs = [executor.submit(
                    self._eval_sampling_op.collate_fn, val_dataloader(bid)) for bid in range(len(val_dataloader))]
            self._test_samples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(torch.get_num_threads()*0.4)) as executor:
                self._test_sampling_jobs = [executor.submit(
                    self._eval_sampling_op.collate_fn, test_dataloader(bid)) for bid in range(len(test_dataloader))]
            self._processed_feature = x

    def generate_taus(self, T):
        self._taus = []
        k = 0
        total_taus = 0
        while total_taus < T:
            tau_i = int(self._tau * np.power(self._rho, k))
            tau_i = min(tau_i, T - total_taus)
            self._taus.append(tau_i)
            total_taus += tau_i
            k += 1 
        return self._taus

    def model_forward(self, x=None, block=None, use_full=False):
        if use_full is False:
            return self._base_model(x, block)
        else:
            return self._base_model(self._processed_feature, self._processed_block)
        
    def flash_sampling(self, total_iter, dataloader):
        min_iter, max_iter = 1, self._max_threads
        count_iter, max_cycle = 0, max(self._taus)
        pre_cycle = np.asarray(list(itertools.accumulate(self._taus)))
        sampling_func = self._training_sampling_op.collate_fn

        while count_iter < total_iter:
            # adaptively increase the number of sampled subgraphs
            curr_cycle = self._taus[pre_cycle.searchsorted(count_iter, 'right')]
            curr_iter = min_iter + int(curr_cycle / max_cycle * (max_iter - min_iter))
            curr_iter = min(curr_iter, total_iter - count_iter)
            count_iter += curr_iter
        
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                sampling_jobs = [executor.submit(sampling_func, dataloader) for _ in range(curr_iter)]

                for future in concurrent.futures.as_completed(sampling_jobs):
                    yield (future.result())

    def sequential_sampling(self, do_val):
        if do_val is True:
            if len(self._val_samples) == 0:
                # When val_sampling is called at the first time, 
                # it would take a little more time to receive the subgraphs.
                print('Waiting for validation minibatch...')
                # Order won't be the same, but it doesn't matter
                for future in concurrent.futures.as_completed(self._val_sampling_jobs):
                    self._val_samples.append(future.result())
                print('Validation minibatch is ready...')

            return self._val_samples
        else:
            if len(self._test_samples) == 0:
                print('Waiting for test minibatch...')
                for future in concurrent.futures.as_completed(self._test_sampling_jobs):
                    self._test_samples.append(future.result())
                print('Test minibatch is ready...')
            
            return self._test_samples

        
