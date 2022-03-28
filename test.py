import torch
from dataset.reddit import Reddit

from models.SGAP_models import GAMLPRecursive, GAMLP, SGC, SIGN
from dataset.planetoid import Planetoid
from dataset.ogbn import Ogbn
from tasks.node_classification import NodeClassification
from dataset.ppi import PPI
from dataset.facebook import Facebook
from dataset.twitch import Twitch
from dataset.karateclub import KarateClub
from dataset.airports import Airports

#dataset = Planetoid("cora", "./", "official")
#dataset = Ogbn("products", "./", "official")
dataset = Airports()

model = SIGN(prop_steps=6, feat_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, num_layers=2)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
us_airports_test = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-4, epochs=200, device=device)
