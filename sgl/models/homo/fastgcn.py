from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.sampler import FastGCNSampler
from sgl.models.simple_models import GCN


class FastGCN(BaseSAMPLEModel):
    def __init__(self, dataset, hidden_dim, output_dim, dropout=0.5, layer_sizes="128-128", prob_type="normalize_col", inductive=True, device="cpu"):
        super(FastGCN, self).__init__(evaluate_mode="full")
        layer_sizes = layer_sizes.split("-")
        layer_sizes = [int(layer_size) for layer_size in layer_sizes]
        self._pre_graph_op = LaplacianGraphOp(r=0.5)
        # inductive-learning
        self._sampling_op = FastGCNSampler(
            self._pre_graph_op._construct_adj(
                dataset.adj[dataset.train_idx, :][:, dataset.train_idx]
            ) if inductive is True else self._pre_graph_op._construct_adj(
                dataset.adj),
            layer_sizes=layer_sizes,
            prob_type=prob_type
        )
        self._base_model = GCN(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=output_dim, nlayers=len(layer_sizes), dropout=dropout
        ).to(device)
