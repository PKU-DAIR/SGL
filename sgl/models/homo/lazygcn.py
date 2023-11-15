from sgl.sampler.utils import RandomBatch
from sgl.models.simple_models import RecycleGCN
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.utils import sparse_mx_to_torch_sparse_tensor

import torch
import numpy as np
import concurrent.futures

class LazyGCN(BaseSAMPLEModel):
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, train_batch_size, dropout=0.5, num_layers=2, max_workers=5, max_threads=-1, rho=1.1, tau=2, num_iters=1, device="cpu"):
        super(LazyGCN, self).__init__()
        self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._max_workers = max_workers
        self._max_threads = max_threads if max_threads > -1 else torch.get_num_threads() // 2
        self._device = device
        # hyperparameters for recycling
        self._rho = rho
        self._tau = tau 
        self._num_iters = num_iters
        self._minibatch = RandomBatch(dataset.train_idx, train_batch_size)
        # define the base model
        self._base_model = RecycleGCN(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)

    def preprocess(self, adj, x):
        self._norm_adj = self._pre_graph_op._construct_adj(adj)
        self._norm_adj = sparse_mx_to_torch_sparse_tensor(self._norm_adj).to(self._device)
        self._processed_feature = x.to(self._device)

    def generate_taus(self, T):
        taus = []
        k = 0
        total_taus = 0
        while total_taus < T:
            tau_i = int(self._tau * np.power(self._rho, k))
            tau_i = min(tau_i, T - total_taus)
            taus.append(tau_i)
            total_taus += tau_i
            k += 1 
        
        return taus

    def flash_sampling(self, num_iter):
        num_loops = (num_iter + self._max_threads - 1) // self._max_threads
        num_iter_pt = self._max_threads
        sampling_func = self._training_sampling_op.sampling
   
        for l in range(num_loops):
            
            if l == num_loops - 1:
                num_iter_pt = num_iter - ((num_loops - 1) * self._max_threads)
        
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                sampling_jobs = [executor.submit(sampling_func, self._minibatch) for _ in range(num_iter_pt)]

                for future in concurrent.futures.as_completed(sampling_jobs):
                    yield (future.result())
