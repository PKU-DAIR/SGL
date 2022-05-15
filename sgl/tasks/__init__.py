from .node_classification import HeteroNodeClassification
from .node_classification import NodeClassification
from .node_clustering import NodeClustering
from .node_clustering import NodeClusteringNAFS
from .link_prediction import LinkPredictionGAE
from .link_prediction import LinkPredictionNAFS
from .correct_and_smooth import NodeClassification_With_CorrectAndSmooth

__all__ = [
    "NodeClassification",
    "HeteroNodeClassification",
    "NodeClustering",
    "NodeClusteringNAFS",
    "LinkPredictionGAE",
    "LinkPredictionNAFS",
    "NodeClassification_With_CorrectAndSmooth"
]
