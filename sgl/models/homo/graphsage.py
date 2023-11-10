from sgl.sampler import NeighborSampler
from sgl.models.simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp


class GraphSAGE(BaseSAMPLEModel):
    def __init__(self, dataset, hidden_dim, output_dim, dropout=0.5, inductive=False, layer_sizes="20-10", device="cpu"):
        super(GraphSAGE, self).__init__(evaluate_mode="full")
        layer_sizes = layer_sizes.split("-")
        layer_sizes = [int(layer_size) for layer_size in layer_sizes]
        self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._sampling_op = NeighborSampler(
            self._pre_graph_op._construct_adj(
                dataset.adj[dataset.train_idx, :][:, dataset.train_idx] if inductive else dataset.adj
            ),
            layer_sizes=layer_sizes,
        )
        self._base_model = GCN(
            nfeat=dataset.num_features, nhid=hidden_dim, nclass=output_dim, nlayers=len(layer_sizes), dropout=dropout
        ).to(device)
