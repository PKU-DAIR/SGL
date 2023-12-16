import sgl.models.simple_models as SimpleModels
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp, RwGraphOp


class VanillaGNN(BaseSAMPLEModel):
    """
    It is a naive version of Graph Convolutional Network which works in full-batch training.
    """
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, basemodel="GCN", sparse_type="torch", dropout=0.5, num_layers=2, device="cpu"):
        super(VanillaGNN, self).__init__(evaluate_mode="full", sparse_type=sparse_type)
        if basemodel == "SAGE":
            self._pre_graph_op = RwGraphOp()
        elif basemodel == "GCN":
            self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = getattr(SimpleModels, basemodel)(
            n_feat=dataset.num_features, n_hid=hidden_dim, n_class=dataset.num_classes, n_layers=num_layers, dropout=dropout
        ).to(device)
