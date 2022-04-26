## SGL: Scalable Graph Learning

**SGL** is a Graph Neural Network (GNN) toolkit targeting scalable graph learning, which supports deep graph learning on
extremely large datasets. SGL allows users to easily implement scalable graph neural networks and evaluate its
performance on various downstream tasks like node classification, node clustering, and link prediction. Further, SGL
supports auto neural architecture search functionality based
on <a href="https://github.com/PKU-DAIR/open-box" target="_blank" rel="nofollow">OpenBox</a>. SGL is designed and
developed by the graph learning team from
the <a href="https://cuibinpku.github.io/index.html" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.

## Library Highlights

+ **High scalability**: Follow the scalable design paradigm **SGAP**
  in <a href="https://arxiv.org/abs/2203.00638" target="_blank" rel="nofollow">PaSca</a>, SGL scale to graph data with
  billions of nodes and edges.
+ **Auto neural architecture search**: Automatically choose decent neural architectures according to specific tasks, and
  pre-defined objectives (e.g., inference time).
+ **Ease of use**: User-friendly interfaces of implementing existing scalable GNNs and executing various downstream
  tasks.

## Installation

Some datasets in SGL are constructed based
on <a href="https://github.com/pyg-team/pytorch_geometric" target="_blank" rel="nofollow">PyG</a>. Please follow the
link below to install PyG first before installing
SGL: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

### Install from pip

To install SGL from PyPI:

```bash
pip install sgl-dair
```

## Quick Start

A quick start example is given by:

```python
from SGL.dataset import Planetoid
from SGL.models.homo import SGC
from SGL.tasks import NodeClassification

dataset = Planetoid("pubmed", "./", "official")
model = SGC(prop_steps=3, feat_dim=dataset.num_features, num_classes=dataset.num_classes)

device = "cuda:0"
test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-5, epochs=200, device=device).test_acc
```

An example of the auto neural network search functionality is as follows:

```python
import torch
from SGL.dataset.planetoid import Planetoid
from SGL.search.search_config import ConfigManager
from openbox.optimizer.generic_smbo import SMBO

dataset = Planetoid("cora", "./", "official")
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

initial_arch = [2, 0, 1, 2, 3, 0, 0]
configer = ConfigManager(initial_arch)
configer._setParameters(dataset, device, 64, 200, 1e-2, 5e-4)

dim = 7
bo = SMBO(configer._configFunction,
          configer._configSpace(),
          num_objs=2,
          num_constraints=0,
          max_runs=3500,
          surrogate_type='prf',
          acq_type='ehvi',
          acq_optimizer_type='local_random',
          initial_runs=2*(dim+1),
          init_strategy='sobol',
          ref_point=[-1, 0.00001],
          time_limit_per_trial=5000,
          task_id='quick_start',
          random_state=1)
history = bo.run()
print(history)
```
The distributed version of the graph neural architecture search is built on the code used by our cooperation partner Tencent, and we will release this part ASAP.

## Related Publications

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm** Wentao Zhang, Yu Shen, Zheyu Lin, Yang
Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui; The world wide web conference (WWW 2022, CCF-A)
. https://arxiv.org/abs/2203.00638

## License

The entire codebase is under [MIT license](LICENSE).
