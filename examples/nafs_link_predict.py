from sgl.dataset import Planetoid
from sgl.tasks import LinkPredictionNAFS

dataset = Planetoid("pubmed", "./", "official")

device = "cuda:0"
best_hop_roc_auc, best_hop_avg_prev, best_roc_auc, best_avg_prec = LinkPredictionNAFS(dataset)
