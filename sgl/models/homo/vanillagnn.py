import sgl.models.simple_models as SimpleModels
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp, RwGraphOp
from sgl.tasks.utils import sparse_mx_to_torch_sparse_tensor


class VanillaGNN(BaseSAMPLEModel):
    """
    It is a naive version of Graph Convolutional Network which works in full-batch training.
    """
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, basemodel="GCN", dropout=0.5, num_layers=2, device="cpu"):
        super(VanillaGNN, self).__init__(evaluate_mode="full")
        if basemodel == "SAGE":
            self._pre_graph_op = RwGraphOp()
        elif basemodel == "GCN":
            self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = getattr(SimpleModels, basemodel)(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=dataset.num_classes, nlayers=num_layers, dropout=dropout
        ).to(device)

    def preprocess(self, adj, x):
        self._norm_adj = self._pre_graph_op._construct_adj(adj)
        self._norm_adj = sparse_mx_to_torch_sparse_tensor(self._norm_adj)
        self._processed_feature = x
