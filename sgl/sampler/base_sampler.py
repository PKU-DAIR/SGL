class BaseSampler:
    def __init__(self, adj, **kwargs):
        self.adj = adj
        self.sampler_name = "None"
        self._post_sampling_op = None
        self.pre_sampling = False
        
        self._pre_process(**kwargs)

    def _pre_process(self, **kwargs):
        pass

    def sampling(self, batch_inds):
        raise NotImplementedError
    
    def _post_process(self, adjs, to_sparse_tensor=True):
        raise NotImplementedError
