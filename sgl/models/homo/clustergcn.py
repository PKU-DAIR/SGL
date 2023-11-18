from sgl.models.simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel

class ClusterGCN(BaseSAMPLEModel):
    def __init__(self, training_eval_sampler, nfeat, hidden_dim, nclass, dropout=0.5, num_layers=2, device="cpu"):
        super(ClusterGCN, self).__init__(evaluate_mode="sampling")
        self._training_sampling_op = training_eval_sampler
        self._eval_sampling_op = training_eval_sampler
        self._base_model = GCN(nfeat=nfeat, nhid=hidden_dim, nclass=nclass, nlayers=num_layers, dropout=dropout).to(device)

    def sampling(self, batch_inds):      
        if self.training:
            return self._training_sampling_op.sampling(batch_inds, training=True)
        else:
            return self._eval_sampling_op.sampling(batch_inds, training=False)