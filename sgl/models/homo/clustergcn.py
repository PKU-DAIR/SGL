from sgl.sampler import ClusterGCNSampler
from sgl.models.simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import LaplacianGraphOp

class ClusterGCN(BaseSAMPLEModel):
    def __init__(self, adj, features, target, device, nfeat, nhid, nclass, clustering_method="random", cluster_number=32, test_ratio=0.3):
        super(ClusterGCN, self).__init__(evaluate_mode="sampling")
        self._pre_graph_op = LaplacianGraphOp(r=0.5)
        self._sampling_op = ClusterGCNSampler(adj, features, target, clustering_method=clustering_method, cluster_number=cluster_number, test_ratio=test_ratio)
        self._base_model = GCN(nfeat, nhid, nclass).to(device)