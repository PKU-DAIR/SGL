import torch

from sgl.dataset import OgbnMag
from sgl.models.hetero import NARS_SIGN
from sgl.tasks import HeteroNodeClassification
from sgl.utils import GpuWithMaxFreeMem

dataset = OgbnMag("mag", "./")
predict_class = 'paper'
random_subgraph_num = 2
subgraph_edge_type_num = 1
model = NARS_SIGN(prop_steps=3, feat_dim=dataset.data.num_features[predict_class],
                  output_dim=dataset.data.num_classes[predict_class], hidden_dim=256, num_layers=2,
                  random_subgraph_num=random_subgraph_num, subgraph_edge_type_num=subgraph_edge_type_num)

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
test_acc = HeteroNodeClassification(dataset, predict_class, model, lr=0.001, weight_decay=0, epochs=200, device=device,
                                    train_batch_size=10000, eval_batch_size=10000).test_acc
