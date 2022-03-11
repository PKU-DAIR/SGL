import torch

from models.SGAP_models import GAMLPRecursive, GAMLP, SGC, SIGN
from dataset.planetoid import Planetoid
from dataset.ogbn import Ogbn
from tasks.node_classification import NodeClassification

dataset = Planetoid("pubmed", "./", "official")
# dataset = Ogbn("products", "./", "official")

model = SIGN(prop_steps=6, feat_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, num_layers=2)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
cora_test = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-4, epochs=200, device=device)
