from sgl.sampler import NeighborSampler
from sgl.models.simple_models import SAGE
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import RwGraphOP
from sgl.operators.message_op import PreNormMessageOp

class GraphSAGE(BaseSAMPLEModel):
    def __init__(self, dataset, sampler, hidden_dim, dropout=0.5, num_layers=2, device="cpu"):
        super(GraphSAGE, self).__init__(evaluate_mode="full")
        self._pre_graph_op = RwGraphOP()
        self._pre_feature_op = PreNormMessageOp(p=1, dim=1)
        self._sampling_op = sampler
        self._post_sampling_graph_op = RwGraphOP()
        self._base_model = SAGE(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)
