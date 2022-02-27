import torch

from models.SGAP_models import SGC
from dataset.planetoid import Planetoid
from tasks.node_classification import NodeClassification

dataset = Planetoid("cora", "./", "official")

model = SGC(prop_steps=3, feat_dim=dataset.num_features, num_classes=dataset.num_classes)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
cora_test = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-3, epochs=200, device=device)
