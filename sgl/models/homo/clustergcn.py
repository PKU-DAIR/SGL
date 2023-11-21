from sgl.models.simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp

class ClusterGCN(BaseSAMPLEModel):
    def __init__(self, training_sampler, eval_sampler, nfeat, hidden_dim, nclass, dropout=0.5, num_layers=2, device="cpu"):
        super(ClusterGCN, self).__init__(evaluate_mode="sampling")
        self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = GCN(nfeat=nfeat, nhid=hidden_dim, nclass=nclass, nlayers=num_layers, dropout=dropout).to(device)