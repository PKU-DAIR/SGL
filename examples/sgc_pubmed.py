from sgl.dataset import Planetoid
from sgl.models.homo import SGC
from sgl.tasks import NodeClassification

dataset = Planetoid("pubmed", "./", "official")
model = SGC(prop_steps=3, feat_dim=dataset.num_features, output_dim=dataset.num_classes)

device = "cuda:0"
test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-5, epochs=200, device=device).test_acc
