import torch

from models.SGAP_models import GAMLP
from dataset.planetoid import Planetoid
from tasks.node_classification import NodeClassification

dataset = Planetoid("cora", "./", "official")

model = GAMLP(prop_steps=12, feat_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=32,
              num_layers=2)
model.preprocess(dataset.adj, dataset.x)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
cora_test = NodeClassification((dataset.y, dataset.train_idx, dataset.val_idx, dataset.test_idx), model, lr=0.01,
                               weight_decay=5e-6, epochs=200, device=device)
