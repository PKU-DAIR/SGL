## SGL: Scalable Graph Learning

**SGL** is a Graph Neural Network (GNN) toolkit targeting scalable graph learning, which supports deep graph learning on
extremely large datasets. SGL allows users to easily implement scalable graph neural networks and evaluate its
performance on various downstream tasks like node classification, node clustering, and link prediction. Further, SGL
supports auto neural architecture search functionality based
on <a href="https://github.com/PKU-DAIR/open-box" target="_blank" rel="nofollow">OpenBox</a>. SGL is designed and
developed by the graph learning team from
the <a href="https://cuibinpku.github.io/index.html" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.

## Why SGL？
The key difference between SGL and existing GNN toolkits, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), is that, SGL enjoys the characteristics of the follwing three perspectives.

+ **High scalability**: Following the scalable design paradigm **SGAP**
  in <a href="https://arxiv.org/abs/2203.00638" target="_blank" rel="nofollow">PaSca</a>, SGL can scale to graph data with
  billions of nodes and edges. 
+ **Auto neural architecture search**: SGL can automatically choose decent and scalable graph neural architectures according to specific tasks and
  pre-defined multiple objectives (e.g., inference time, memory cost, and predictive performance).
+ **Ease of use**: SGL has user-friendly interfaces for implementing existing scalable GNNs and executing various downstream tasks.

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
from sgl.dataset import Planetoid
from sgl.models.homo import SGC
from sgl.tasks import NodeClassification

dataset = Planetoid("pubmed", "./", "official")
model = SGC(prop_steps=3, feat_dim=dataset.num_features, output_dim=dataset.num_classes)

device = "cuda:0"
test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-5, epochs=200, device=device).test_acc
```

An example of the auto neural network search functionality is as follows:

```python
import torch
from openbox.optimizer.generic_smbo import SMBO

from sgl.dataset.planetoid import Planetoid
from sgl.search.search_config import ConfigManager

dataset = Planetoid("cora", "./", "official")
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

## Define Initial Arch and Configuration
initial_arch = [2, 0, 1, 2, 3, 0, 0]
configer = ConfigManager(initial_arch)
configer._setParameters(dataset, device, 128, 200, 1e-2, 5e-4)

## Define Search Parameters
dim = 7
bo = SMBO(configer._configFunction,
          configer._configSpace(),
          num_objs=2,
          num_constraints=0,
          max_runs=3500,
          surrogate_type='prf',
          acq_type='ehvi',
          acq_optimizer_type='local_random',
          initial_runs=2 * (dim + 1),
          init_strategy='sobol',
          ref_point=[-1, 0.00001],
          time_limit_per_trial=5000,
          task_id='quick_start',
          random_state=1)

## Search
history = bo.run()
print(history)
```

## Related Publications

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm**[[PDF](https://dl.acm.org/doi/pdf/10.1145/3485447.3511986)]<br>
Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui.<br>
The world wide web conference.<br>
***WWW 2022, CCF-A, 🏆 Best Student Paper Award (among 1822 submmisions)</font></b>***


**Node Dependent Local Smoothing for Scalable Graph Learning** [[PDF](https://arxiv.org/pdf/2110.14377)]<br>
Wentao Zhang, Mingyu Yang, Zeang Sheng, Yang Li, Wen Ouyang, Yangyu Tao, Zhi Yang, Bin Cui.<br>
Thirty-fifth Conference on Neural Information Processing Systems.<br>
***NeurIPS 2021, CCF-A, <font color=orange>Spotlight Presentation, Acceptance Rate: < 3%</font>***. 

**NAFS: A Simple yet Tough-to-beat Baseline for Graph Representation Learning.** [[PDF](https://arxiv.org/abs/2206.08583)]<br>
Wentao Zhang, Zeang Sheng, Mingyu Yang, Yang Li, Yu Shen, Zhi Yang, Bin Cui.<br>
The 39th International Conference on Machine Learning.<br>
***ICML 2022, CCF-A***. 

**Deep and Flexible Graph Neural Architecture Search.** [[PDF](https://arxiv.org/abs/2206.08582)]<br>
Wentao Zhang, Zheyu Lin, Yu Shen, Yang Li, Zhi Yang, Bin Cui.<br>
The 39th International Conference on Machine Learning.<br>
***ICML 2022, CCF-A***. 
  
**Model Degradation Hinders Deep Graph Neural Networks.** [[PDF](https://arxiv.org/abs/2206.04361)]<br>
Wentao Zhang, Zeang Sheng, Yuezihan Jiang, Yikuan Xia, Jun Gao, Zhi Yang, Bin Cui.<br>
SIGKDD Conference on Knowledge Discovery and Data Mining.<br>
***KDD 2022, CCF-A***. 

**Graph Attention Multi-Layer Perceptron** [[PDF](https://arxiv.org/pdf/2108.10097)]<br>
Wentao Zhang, Ziqi Yin, Zeang Sheng, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, Bin Cui.<br>
ACM SIGKDD Conference on Knowledge Discovery and Data Mining. <br>
***KDD 2022, CCF-A, Rank \#1 in [Open Graph Benchmark](https://ogb.stanford.edu/docs/leader_nodeprop/\#ogbn-mag)*** 
  
**[OpenBox](https://github.com/PKU-DAIR/open-box): A Generalized Black-box Optimization Service** [[PDF](https://arxiv.org/abs/2106.00421)]<br>
Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, ..., Wentao Wu, Zhi Yang, Ce Zhang, Bin Cui.<br>
ACM SIGKDD Conference on Knowledge Discovery and Data Mining.<br> 
***KDD 2021, CCF-A, top prize in [open-source innovation competition @ 2021 CCF ChinaSoft](https://mp.weixin.qq.com/s/8JX5ymkUt5MvDcHLOjB3Xw)***



## Citing SGL

Please cite our [paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511986) if you find *SGL* useful in your work:
```
@inproceedings{zhang2022pasca,
  title={PaSca: A Graph Neural Architecture Search System under the Scalable Paradigm},
  author={Zhang, Wentao and Shen, Yu and Lin, Zheyu and Li, Yang and Li, Xiaosen and Ouyang, Wen and Tao, Yangyu and Yang, Zhi and Cui, Bin},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1817--1828},
  year={2022}
}
```

## Contact

If you have any technical questions, please submit new issues.

If you have any other questions, please contact: Wentao Zhang[wentao.zhang@pku.edu.cn] and Zeang Sheng[shengzeang18@pku.edu.cn].
  
## License

The entire codebase is under [MIT license](LICENSE).
