from sgl.models.simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import RwGraphOp

class GraphSAINT(BaseSAMPLEModel):
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, dropout=0.5, num_layers=2, device="cpu"):
        super(GraphSAINT, self).__init__()
        self._pre_graph_op = RwGraphOp()
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = GCN(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)

    @property
    def cur_loss_norm(self):
        return self._training_sampling_op.loss_norm[self._training_sampling_op.index]