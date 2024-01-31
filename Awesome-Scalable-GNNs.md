# Awsome-Scalable-GNNs

## Contents
- [Scalable-GNNs](#Scalable-GNN)
  - [Training Acceleration](#Training)
    - [1.Sampling](#Sampling)
    - [2.Linear Model](#Linear-Model)
  - [Inference Acceleration](#Inference)
    - [1.Knowledge Distillation](#KD)
    - [2.Quantization](#Quantization)
    - [3.Pruning](#Pruning)
  - [Execution Acceleration](#Execution)
    - [1. Binarization](#Binarization)
    - [2. Graph Condensation](#Graph_Condensation)


<a name="Scalable-GNN" />

To scale GNNs to extremely large graphs, existing works can be classified into the following types.
<a name="Training"/>

<a name="Sampling" />

## 1. Sampling
<p class="center">
    <img src="img/GraphSAGE.jpg" width="60%">
    <br>
    <em>Source: Inductive Representation Learning on Large Graphs</em>
</p>

<p class="center">
    <img src="img/Sampling.jpg" width="60%">
    <br>
    <em>Illustration of different graph sampling methods. Red nodes are selected nodes in the current batch as n<sup>(L)</sup>, blue nodes are nodes sampled in the 1st layer as n<sup>(1)</sup> and green nodes are sampled in the 2st layer as n<sup>(0)</sup>. n<sup>(0)</sup> and n<sup>(1)</sup> form Block<sup>(1)</sup>, n<sup>(1)</sup> and n<sup>(2)</sup> form Block<sup>(2)</sup>. The node-wise sampling method samples 2 nodes for each node (e.g. sampling v<sub>1</sub> and v<sub>6</sub> for v<sub>3</sub> in layer 1. The layer-wise sampling method samples 3 nodes for each GNN layer. The graph-wise sampling method samples a sub-graph for all layers.</em>
</p>

### Node-wise sampling

1. **Inductive Representation Learning on Large Graphs** [NIPS 2017] [[paper]](https://arxiv.org/abs/1706.02216) [[code]](https://github.com/twjiang/graphSAGE-pytorch)
2. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems** [KDD 2018] [[paper]](https://arxiv.org/abs/1806.01973)
3. **Stochastic Training of Graph Convolutional Networks with Variance Reduction** [ICML 2018] [[paper]](https://arxiv.org/abs/1710.10568) [[code]](https://github.com/thu-ml/stochastic_gcn)
4. **Blocking-based neighbor sampling for large-scale graph neural networks** [IJCAI 2021] [[paper]](https://www.ijcai.org/proceedings/2021/0455.pdf)
5. **Bandit samplers for training graph neural networks** [NeurIPS 2020]
6. **Performance-adaptive sampling strategy towards fast and accurate graph neural networks** [KDD 2021]
7. **Hierarchical graph transformer with adaptive node sampling** [NeurIPS 2022]

### Layer-wise sampling

1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling** [ICLR 2018]  [[paper]](https://arxiv.org/abs/1801.10247)[[code]](https://github.com/matenure/FastGCN)
2. **Adaptive Sampling Towards Fast Graph Representation Learning** [NeurIPS 2018] [[paper]](https://arxiv.org/abs/1809.05343) [[code_pytorch]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/_deprecated/adaptive_sampling) [[code_tentsor_flow]](https://github.com/huangwb/AS-GCN)
3. **Layer-dependent importance sampling for training deep and large graph convolutional networks** [NeurIPS 2019]
4. **GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks**  [NeurIPS 2023]


### Graph-wise sampling

1. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks** [KDD 2019] [[paper]](https://arxiv.org/abs/1905.07953) [[code]](https://github.com/google-research/google-research/tree/master/cluster_gcn)
2. **GraphSAINT: Graph Sampling Based Inductive Learning Method** [ICLR 2020] [[paper]](https://arxiv.org/abs/1907.04931) [[code]](https://github.com/GraphSAINT/GraphSAINT)
3. **Minimal variance sampling with provable guarantees for fast training of graph neural networks** [KDD 2020]
4. **Gnnautoscale: Scalable and expressive graph neural networks via historical embeddings** [ICLR 2021]
5. **Decoupling the depth and scope of graph neural networks** [NeurIPS 2021]
6. **Ripple walk training: A subgraph-based training framework for large and deep graph neural network** [IJCNN 2021]
7. **LMC: Fast Training of GNNs via Subgraph Sampling with Provable Convergence** [ICLR 2023]


<a name="Linear-Model" />

## 2. Linear Model
### Simple model without attention
<p class="center">
    <img src="img/NDLS.jpg" width="60%">
    <br>
    <em>Source: Node Dependent Local Smoothing for Scalable Graph Learning</em>
</p>

1. **Simplifying Graph Convolutional Networks** [ICML 2019] [[paper]](https://arxiv.org/abs/1902.07153) [[code]](https://github.com/Tiiiger/SGC)
2. **Scalable Graph Neural Networks via Bidirectional Propagation** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2010.15421) [[code]](https://github.com/chennnM/GBP)
3. **SIGN: Scalable Inception Graph Neural Networks** [ICML 2020] [[paper]](https://arxiv.org/abs/2004.11198) [[code]](https://github.com/twitter-research/sign)
4. **Simple Spectral Graph Convolution** [ICLR 2021] [[paper]](https://openreview.net/forum?id=CYO5T-YjWZV) [[code]](https://github.com/allenhaozhu/SSGC)
5. **Approximate graph propagation** [KDD 2021]
6. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank** [ICLR 2018]
7. **Combining Label Propagation and Simple Models out-performs Graph Neural Networks** [ICLR 2020]
8. **Adaptive propagation graph convolutional network** [TNNLS 2020]
9. **Scaling graph neural networks with approximate pagerank** [KDD 2020]
10. **Node Dependent Local Smoothing for Scalable Graph Learning** [NeurIPS 2021] [[paper]](https://arxiv.org/abs/2110.14377) [[code]](https://github.com/zwt233/NDLS)
11. **NAFS: A Simple yet Tough-to-Beat Baseline for Graph Representation Learning** [OpenReview 2022] [[paper]](https://openreview.net/forum?id=dHJtoaE3yRP) [[code]](https://openreview.net/attachment?id=dHJtoaE3yRP&name=supplementary_material)

### Complex model with attention
1. **Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.09376) [[code]](https://github.com/skepsun/SAGN_with_SLE)
2. **Graph Attention Multi-Layer Perceptron** [KDD 2022] [[paper]](https://arxiv.org/abs/2108.10097) [[code]](https://github.com/zwt233/GAMLP)
3. **NAFS: A Simple yet Tough-to-Beat Baseline for Graph Representation Learning** [OpenReview 2022] [[paper]](https://openreview.net/forum?id=dHJtoaE3yRP) [[code]](https://openreview.net/attachment?id=dHJtoaE3yRP&name=supplementary_material)
4. **Pasca: A graph neural architecture search system under the scalable paradigm** [WWW 2020]
5. **Towards deeper graph neural networks** [KDD 2020]
6. **Node-wise Diffusion for Scalable Graph Learning** [WWW 2023]
7. **Scalable decoupling graph neural network with feature-oriented optimization** [VLDB 2023]
8. **Grand+: Scalable graph random neural networks** [WWW 2022]

<a name="Inference"/>
<a name="KD" />

## 3. Knowledge distillation

### GNN2GNN

1. **Distilling knowledge from graph convolutional networks** [CVPR 2020]
2. **Tinygnn: Learning efficient graph neural networks** [KDD 2020]
3. **On representation knowledge distillation for graph neural networks** [TNNLS 2022]
4. **Graph-free knowledge distillation for graph neural networks** [IJCAI 2021]
5. **Knowledge distillation as efficient pre-training: Faster convergence, higher data-efficiency, and better transferability** [CVPR 2022]
6. **Geometric knowledge distillation: Topology compression for graph neural networks** [NeurIPS 2022]
### GNN2MLP
1. **Graph-mlp: Node classification without message passing in graph** [Arxiv 2021]
2. **Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation** [ICLR 2021]
3. **Extract the knowledge of graph neural networks and go beyond it: An effective knowledge distillation framework** [WWW 2021]
4. **Learning mlps on graphs: A unified view of effectiveness, robustness, and efficiency** [ICLR 2022]
5. **VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs** [ICLR 2024]
6. **Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs** [ICML 2023]

## 4. Quantization
1. **Learned low precision graph neural networks** [Arxiv 2009]
2. **Degree-Quant: Quantization-Aware Training for Graph Neural Networks** [ICLR 2020]
3. **Sgquant: Squeezing the last bit on graph neural networks with specialized quantization** [ICTAI 2020]
4. **VQ-GNN: A universal framework to scale up graph neural networks using vector quantization** [NeurIPS 2021]
5. **A<sup>2</sup>Q: Aggregation-Aware Quantization for Graph Neural Networks** [ICLR 2022]
6. **EPQuant: A Graph Neural Network compression approach based on product quantization** [NC 2022]
7. **Low-bit Quantization for Deep Graph Neural Networks with Smoothness-aware Message Propagation** [CIKM 2023]
8. **Haar wavelet feature compression for quantized graph convolutional networks** [TNNLS 2023]

<a name="Pruning" />

## 5. Pruning
1. **A unified lottery ticket hypothesis for graph neural networks** [ICML 2021]
2. **Accelerating Large Scale Real-Time GNN Inference using Channel Pruning** [VLDB 2021]
3. **Inductive Lottery Ticket Learning for Graph Neural Networks** [Openreview 2021]
4. **Early-bird gcns: Graph-network co-optimization towards more efficient gcn training and inference via drawing early-bird lottery tickets** [AAAI 2022]
5. **Searching Lottery Tickets in Graph Neural Networks: A Dual Perspective** [ICLR 2022]
6. **Rethinking Graph Lottery Tickets: Graph Sparsity Matters** [ICLR 2022]
7. **The snowflake hypothesis: Training deep GNN with one node one receptive field** [ArXiv 2023]

<a name="Execution"/>

<a name="Binarization" />

## 6. Binarization
1. **Bi-gcn: Binary graph convolutional network** [CVPR 2021]
2. **Binarized graph neural network** [WWW 2021]
3. **Binary graph neural networks** [CVPR 2021]
4. **Meta-aggregator: Learning to aggregate for 1-bit graph neural networks** [ICCV 2021]
5. **BitGNN: Unleashing the Performance Potential of Binary Graph Neural Networks on GPUs** [ICS 2023]

<a name="Graph_Condensation" />

## 7. Graph Condensation

1. **Graph Condensation for Graph Neural Networks** [ICLR 2021]
2. **Condensing graphs via one-step gradient matching** [KDD 2022]
3. **Graph condensation via receptive field distribution matching** [Arxiv 2022]
4. **Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data** [Arxiv 2023]
5. **Graph Condensation via Eigenbasis Matching** [Arxiv 2023]
6. **Kernel Ridge Regression-Based Graph Dataset Distillation** [KDD 2023]
7. **Graph Condensation for Inductive Node Representation Learning** [ICDE 2024]
8. **Fast Graph Condensation with Structure-based Neural Tangent Kernel** [Arxiv 2023]