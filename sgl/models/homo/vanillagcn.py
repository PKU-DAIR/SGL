from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.models.simple_models import GCN


class VanillaGCN(BaseSAMPLEModel):
    """
    It is a naive version of Graph Convolutional Network which works in full-batch training.
    """
    def __init__(self, dataset, sampler, hidden_dim, dropout=0.5, num_layers=2, device="cpu"):
        super(VanillaGCN, self).__init__(evaluate_mode="full")
        self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._sampling_op = sampler
        self._base_model = GCN(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)
