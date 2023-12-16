from sgl.models.simple_models import SAGE
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.message_op import PreNormMessageOp
from sgl.operators.graph_op import RwGraphOp

class GraphSAGE(BaseSAMPLEModel):
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, sparse_type="torch", dropout=0.5, num_layers=2, device="cpu"):
        super(GraphSAGE, self).__init__(sparse_type=sparse_type)
        self._pre_graph_op = RwGraphOp()
        self._pre_feature_op = PreNormMessageOp(p=1, dim=1)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = SAGE(
            n_feat=dataset.num_features, n_hid=hidden_dim, n_class=dataset.num_classes, n_layers=num_layers, dropout=dropout
        ).to(device)
