class BaseSampler:
    def __init__(self, adj, **kwargs):
        self.adj = adj
        self.sampler_name = "None"
        self.pre_sampling = False
        
        self._preproc(**kwargs)

    def _preproc(self, **kwargs):
        pass

    def sampling(self, batch_inds):
        raise NotImplementedError
