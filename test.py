import torch
from dataset.reddit import Reddit

from dataset.planetoid import Planetoid
from dataset.ogbn import Ogbn
from dataset.coauthor import Coauthor
from dataset.amazon import Amazon

from models.SGAP_models import GAMLPRecursive, GAMLP, SGC, SIGN
from tasks.node_classification import NodeClassification
from dataset.facebook import Facebook
from dataset.twitch import Twitch
from dataset.karateclub import KarateClub
from dataset.airports import Airports

# dataset = Planetoid("pubmed", "./", "official")
# dataset = Ogbn("products", "./", "official")
dataset = Coauthor("phy", "./")
# dataset = Amazon("computers", "./")

model = SGC(prop_steps=3, feat_dim=dataset.num_features, num_classes=dataset.num_classes)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-4, epochs=200, device=device).test_acc
