from sgl.models.simple_models import SAGE
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.message_op import PreNormMessageOp
from sgl.operators.graph_op import RwGraphOp

class GraphSAGE(BaseSAMPLEModel):
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, dropout=0.5, num_layers=2, device="cpu"):
        super(GraphSAGE, self).__init__()
        self._pre_graph_op = RwGraphOp()
        self._pre_feature_op = PreNormMessageOp(p=1, dim=1)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = SAGE(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)
