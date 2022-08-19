from sgl.dataset import Planetoid
from sgl.tasks import NodeClusteringNAFS

dataset = Planetoid("pubmed", "./", "official")

device = "cuda:0"
best_hop_acc, best_hop_nmi, best_hop_adjscore, best_acc, best_nmi, best_adjscore = NodeClusteringNAFS(dataset)
