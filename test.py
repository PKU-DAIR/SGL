import torch

from models.SGAP_models import GAMLPRecursive, GAMLP, SGC
from dataset.planetoid import Planetoid
from dataset.ogbn import Ogbn
from tasks.node_classification import NodeClassification

# dataset = Planetoid("pubmed", "./", "official")
dataset = Ogbn("products", "./", "official")

model = GAMLP(prop_steps=6, feat_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=512, num_layers=3)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
cora_test = NodeClassification(dataset, model, lr=0.01, weight_decay=0, epochs=200, device=device, train_batch_size=1000, eval_batch_size=1000)
