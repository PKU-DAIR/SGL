# Awsome-GNN-Acceleration

## Contents
- [GNN-Acceleration](#GNN-Acceleration)
  - [Training Acceleration](#Training)
    - [1.Sampling](#Sampling)
    - [2.Linear Model](#Linear-Model)
  - [Inference Acceleration](#Inference)
    - [3.Knowledge Distillation](#KD)
    - [4.Quantization](#Quantization)
    - [5.Pruning](#Pruning)
  - [Execution Acceleration](#Execution)
    - [6. Binarization](#Binarization)
    - [7. Graph Condensation](#Graph_Condensation)


<a name="GNN-Acceleration" />

To scale GNNs to extremely large graphs, existing works can be classified into the following types.
<a name="Training"/>

<a name="Sampling" />

## 1. Sampling
<p class="center">
    <img src="img/Sampling.jpg" width="80%">
    <br>
    <em>Illustration of different graph sampling methods. Red nodes are selected nodes in the current batch as n<sup>(L)</sup>, blue nodes are nodes sampled in the 1st layer as n<sup>(1)</sup> and green nodes are sampled in the 2st layer as n<sup>(0)</sup>. n<sup>(0)</sup> and n<sup>(1)</sup> form Block<sup>(1)</sup>, n<sup>(1)</sup> and n<sup>(2)</sup> form Block<sup>(2)</sup>. The node-wise sampling method samples 2 nodes for each node (e.g. sampling v<sub>1</sub> and v<sub>6</sub> for v<sub>3</sub> in layer 1. The layer-wise sampling method samples 3 nodes for each GNN layer. The graph-wise sampling method samples a sub-graph for all layers.</em>
</p>

### Node-wise sampling

1. **Inductive Representation Learning on Large Graphs** [NIPS 2017] [[paper]](https://arxiv.org/abs/1706.02216) [[code]](https://github.com/twjiang/graphSAGE-pytorch)
2. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems** [KDD 2018] [[paper]](https://arxiv.org/abs/1806.01973)[[code]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage)
3. **Stochastic Training of Graph Convolutional Networks with Variance Reduction** [ICML 2018] [[paper]](https://arxiv.org/abs/1710.10568) [[code]](https://github.com/thu-ml/stochastic_gcn)
4. **Blocking-based neighbor sampling for large-scale graph neural networks** [IJCAI 2021] [[paper]](https://www.ijcai.org/proceedings/2021/0455.pdf)
5. **Bandit samplers for training graph neural networks** [NeurIPS 2020][[paper]](https://proceedings.neurips.cc/paper_files/paper/2020/file/4cea2358d3cc5f8cd32397ca9bc51b94-Paper.pdf)[[code]](https://github.com/xavierzw/gnn-bs)
6. **Performance-adaptive sampling strategy towards fast and accurate graph neural networks** [KDD 2021][[paper]](https://dl.acm.org/doi/pdf/10.1145/3447548.3467284)[[code]](https://github.com/linkedin/PASS-GNN)
7. **Hierarchical graph transformer with adaptive node sampling** [NeurIPS 2022][[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/854a9ab0f323b841955e70ca383b27d1-Paper-Conference.pdf)[[code]](https://github.com/zaixizhang/ANS-GT)

### Layer-wise sampling

1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling** [ICLR 2018]  [[paper]](https://arxiv.org/abs/1801.10247)[[code]](https://github.com/matenure/FastGCN)
2. **Adaptive Sampling Towards Fast Graph Representation Learning** [NeurIPS 2018] [[paper]](https://arxiv.org/abs/1809.05343) [[code_pytorch]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/_deprecated/adaptive_sampling) [[code_tentsor_flow]](https://github.com/huangwb/AS-GCN)
3. **Layer-dependent importance sampling for training deep and large graph convolutional networks** [NeurIPS 2019][[paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/91ba4a4478a66bee9812b0804b6f9d1b-Paper.pdf)[[code]](https://github.com/acbull/LADIES)
4. **GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks**  [NeurIPS 2023][[paper]](https://openreview.net/pdf?id=1JkgXzKKdo)[[code]](https://github.com/dfdazac/grapes)

### Graph-wise sampling

1. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks** [KDD 2019] [[paper]](https://arxiv.org/abs/1905.07953) [[code]](https://github.com/google-research/google-research/tree/master/cluster_gcn)
2. **GraphSAINT: Graph Sampling Based Inductive Learning Method** [ICLR 2020] [[paper]](https://arxiv.org/abs/1907.04931) [[code]](https://github.com/GraphSAINT/GraphSAINT)
3. **Large-Scale Learnable Graph Convolutional Networks** [KDD 2018][[paper]](https://dl.acm.org/doi/abs/10.1145/3219819.3219947)[[code]](https://github.com/divelab/lgcn)
4. **Minimal variance sampling with provable guarantees for fast training of graph neural networks** [KDD 2020][[paper]](https://arxiv.org/pdf/2006.13866.pdf)[[code]](https://github.com/CongWeilin/mvs_gcn)
5. **Gnnautoscale: Scalable and expressive graph neural networks via historical embeddings** [ICLR 2021] [[paper]](https://arxiv.org/pdf/2106.05609.pdf)[[code]](https://github.com/rusty1s/pyg_autoscale)
6. **Decoupling the depth and scope of graph neural networks** [NeurIPS 2021][[paper]](https://openreview.net/pdf?id=_IY3_4psXuf)[[code]](https://github.com/facebookresearch/shaDow_GNN)
7. **Ripple walk training: A subgraph-based training framework for large and deep graph neural network** [IJCNN 2021] [[paper]](https://arxiv.org/pdf/2002.07206.pdf)[[code]](https://github.com/anonymous2review/RippleWalk)
8. **LMC: Fast Training of GNNs via Subgraph Sampling with Provable Convergence** [ICLR 2023]  [[paper]](https://openreview.net/pdf?id=5VBBA91N6n)[[code]](https://github.com/MIRALab-USTC/GNN-LMC)


<a name="Linear-Model" />

## 2. Linear Model

<p class="center">
    <img src="img/SGAP.png" width="80%">
    <br>
    <em>Illustration of SGAP for Linear model</em>
</p>

### Simple model without attention

1. **Simplifying Graph Convolutional Networks** [ICML 2019] [[paper]](https://arxiv.org/abs/1902.07153) [[code]](https://github.com/Tiiiger/SGC)
2. **Scalable Graph Neural Networks via Bidirectional Propagation** [NeurIPS 2020] [[paper]](https://arxiv.org/abs/2010.15421) [[code]](https://github.com/chennnM/GBP)
3. **SIGN: Scalable Inception Graph Neural Networks** [ICML 2020] [[paper]](https://arxiv.org/abs/2004.11198) [[code]](https://github.com/twitter-research/sign)
4. **Simple Spectral Graph Convolution** [ICLR 2021] [[paper]](https://openreview.net/forum?id=CYO5T-YjWZV) [[code]](https://github.com/allenhaozhu/SSGC)
5. **Approximate graph propagation** [KDD 2021]  [[paper]](https://arxiv.org/pdf/2106.03058.pdf)[[code]](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)
6. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank** [ICLR 2018]  [[paper]](https://openreview.net/pdf?id=H1gL-2A9Ym)[[code]](https://github.com/benedekrozemberczki/APPNP)
7. **Combining Label Propagation and Simple Models out-performs Graph Neural Networks** [ICLR 2020] [[paper]](https://openreview.net/pdf?id=8E1-f3VhX1o)[[code]](https://github.com/CUAI/CorrectAndSmooth)
8. **Adaptive propagation graph convolutional network** [TNNLS 2020] [[paper]](https://arxiv.org/pdf/2002.10306.pdf)[[code]](https://github.com/spindro/AP-GCN)
9. **Scaling graph neural networks with approximate pagerank** [KDD 2020] [[paper]](https://arxiv.org/pdf/2007.01570.pdf)[[code]](https://github.com/TUM-DAML/pprgo_pytorch)
10. **Node Dependent Local Smoothing for Scalable Graph Learning** [NeurIPS 2021] [[paper]](https://arxiv.org/abs/2110.14377) [[code]](https://github.com/zwt233/NDLS)
11. **NAFS: A Simple yet Tough-to-Beat Baseline for Graph Representation Learning** [ICML 2022] [[paper]](https://openreview.net/forum?id=dHJtoaE3yRP) [[code]](https://openreview.net/attachment?id=dHJtoaE3yRP&name=supplementary_material)

### Complex model with attention
1. **Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training** [Arxiv 2021] [[paper]](https://arxiv.org/abs/2104.09376) [[code]](https://github.com/skepsun/SAGN_with_SLE)
2. **Graph Attention Multi-Layer Perceptron** [KDD 2022] [[paper]](https://arxiv.org/abs/2108.10097) [[code]](https://github.com/zwt233/GAMLP)
3. **Pasca: A graph neural architecture search system under the scalable paradigm** [WWW 2022] [[paper]](https://arxiv.org/pdf/2203.00638.pdf)[[code]](https://github.com/PKU-DAIR/SGL)
4. **Towards deeper graph neural networks** [KDD 2020] [[paper]](https://arxiv.org/pdf/2007.09296.pdf)[[code]](https://github.com/mengliu1998/DeeperGNN)
5. **Node-wise Diffusion for Scalable Graph Learning** [WWW 2023] [[paper]](https://arxiv.org/pdf/2305.14000.pdf)[[code]](https://github.com/kkhuang81/NIGCN)
6. **Scalable decoupling graph neural network with feature-oriented optimization** [VLDB 2023] [[paper]](https://arxiv.org/pdf/2207.09179.pdf)[[code]](https://github.com/gdmnl/SCARA-PPR)
7. **Grand+: Scalable graph random neural networks** [WWW 2022] [[paper]](https://arxiv.org/pdf/2203.06389.pdf)[[code]](https://github.com/THUDM/GRAND)

<a name="Inference"/>
<a name="KD" />

## 3. Knowledge distillation

### GNN2GNN

1. **Distilling knowledge from graph convolutional networks** [CVPR 2020] [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Distilling_Knowledge_From_Graph_Convolutional_Networks_CVPR_2020_paper.pdf)[[code]](https://github.com/ihollywhy/DistillGCN.PyTorch)
2. **Tinygnn: Learning efficient graph neural networks** [KDD 2020] [[paper]](https://dl.acm.org/doi/10.1145/3394486.3403236)[[code]]()
3. **On representation knowledge distillation for graph neural networks** [TNNLS 2022] [[paper]](https://arxiv.org/pdf/2111.04964.pdf)[[code]](https://github.com/chaitjo/efficient-gnns)
4. **Graph-free knowledge distillation for graph neural networks** [IJCAI 2021] [[paper]](https://arxiv.org/pdf/2105.07519.pdf)[[code]](https://github.com/Xiang-Deng-DL/GFKD)
5. **Knowledge distillation as efficient pre-training: Faster convergence, higher data-efficiency, and better transferability** [CVPR 2022] [[paper]](https://arxiv.org/pdf/2203.05180.pdf)[[code]](https://github.com/CVMI-Lab/KDEP)
6. **Geometric knowledge distillation: Topology compression for graph neural networks** [NeurIPS 2022] [[paper]](https://openreview.net/pdf?id=7WGNT3MHyBm)[[code]](https://github.com/chr26195/GKD)
### GNN2MLP
1. **Graph-mlp: Node classification without message passing in graph** [Arxiv 2021] [[paper]](https://arxiv.org/pdf/2106.04051.pdf)[[code]](https://github.com/yanghu819/Graph-MLP)
2. **Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation** [ICLR 2021] [[paper]](https://arxiv.org/pdf/2110.08727.pdf)[[code]](https://github.com/snap-research/graphless-neural-networks)
3. **Extract the knowledge of graph neural networks and go beyond it: An effective knowledge distillation framework** [WWW 2021] [[paper]](https://arxiv.org/pdf/2103.02885.pdf)[[code]](https://github.com/BUPT-GAMMA/CPF)
4. **Learning mlps on graphs: A unified view of effectiveness, robustness, and efficiency** [ICLR 2022] [[paper]](https://openreview.net/pdf?id=Cs3r5KLdoj)[[code]](https://github.com/meettyj/NOSMOG)
5. **VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs** [ICLR 2024] [[paper]](https://arxiv.org/pdf/2308.02117.pdf)[[code]](https://github.com/YangLing0818/VQGraph)
6. **Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs** [ICML 2023] [[paper]](https://arxiv.org/pdf/2306.05628.pdf)[[code]](https://github.com/LirongWu/KRD)
7. **Propagate & Distill: Towards Effective Graph Learners Using Propagation-Embracing MLPs** [[paper]](https://openreview.net/pdf?id=2A14hhZsnA) 

<a name="Quantization"/>

## 4. Quantization
1. **Learned low precision graph neural networks** [Arxiv 2009] [[paper]](https://arxiv.org/pdf/2009.09232.pdf)
2. **Degree-Quant: Quantization-Aware Training for Graph Neural Networks** [ICLR 2020] [[paper]](https://arxiv.org/pdf/2008.05000.pdf)[[code]](https://github.com/camlsys/degree-quant)
3. **Sgquant: Squeezing the last bit on graph neural networks with specialized quantization** [ICTAI 2020] [[paper]](https://arxiv.org/pdf/2007.05100.pdf)[[code]](https://github.com/YukeWang96/SGQuant)
4. **VQ-GNN: A universal framework to scale up graph neural networks using vector quantization** [NeurIPS 2021] [[paper]](https://arxiv.org/pdf/2110.14363.pdf)[[code]](https://github.com/devnkong/VQ-GNN)
5. **A<sup>2</sup>Q: Aggregation-Aware Quantization for Graph Neural Networks** [ICLR 2022] [[paper]](https://arxiv.org/pdf/2302.00193.pdf)[[code]](https://github.com/weihai-98/A-2Q)
6. **EPQuant: A Graph Neural Network compression approach based on product quantization** [NC 2022] [[paper]](https://dl.acm.org/doi/10.1016/j.neucom.2022.06.097)
7. **Low-bit Quantization for Deep Graph Neural Networks with Smoothness-aware Message Propagation** [CIKM 2023] [[paper]](https://arxiv.org/pdf/2308.14949v1.pdf)
8. **Haar wavelet feature compression for quantized graph convolutional networks** [TNNLS 2023] [[paper]](https://arxiv.org/pdf/2110.04824.pdf)

<a name="Pruning" />

## 5. Pruning
1. **A unified lottery ticket hypothesis for graph neural networks** [ICML 2021] [[paper]](https://arxiv.org/pdf/2102.06790.pdf)[[code]](https://github.com/VITA-Group/Unified-LTH-GNN)
2. **Accelerating Large Scale Real-Time GNN Inference using Channel Pruning** [VLDB 2021] [[paper]](https://arxiv.org/pdf/2105.04528.pdf)[[code]](https://github.com/tedzhouhk/GCNP)
3. **Inductive Lottery Ticket Learning for Graph Neural Networks** [Openreview 2021] [[paper]](https://openreview.net/pdf?id=Bel1Do_eZC)[[code]](https://github.com/yongduosui/ICPG)
4. **Early-bird gcns: Graph-network co-optimization towards more efficient gcn training and inference via drawing early-bird lottery tickets** [AAAI 2022] [[paper]](https://arxiv.org/pdf/2103.00794.pdf)[[code]](https://github.com/GATECH-EIC/Early-Bird-GCN)
5. **Searching Lottery Tickets in Graph Neural Networks: A Dual Perspective** [ICLR 2022] [[paper]](https://openreview.net/pdf?id=Dvs-a3aymPe)
6. **Rethinking Graph Lottery Tickets: Graph Sparsity Matters** [ICLR 2022] [[paper]](https://openreview.net/pdf?id=fjh7UGQgOB)
7. **The snowflake hypothesis: Training deep GNN with one node one receptive field** [ArXiv 2023] [[paper]](https://arxiv.org/pdf/2308.10051.pdf)

<a name="Execution"/>

<a name="Binarization" />

## 6. Binarization
1. **Bi-gcn: Binary graph convolutional network** [CVPR 2021] [[paper]](https://arxiv.org/pdf/2010.07565.pdf)[[code]](https://github.com/bywmm/Bi-GCN)
2. **Binarized graph neural network** [WWW 2021] [[paper]](https://arxiv.org/pdf/2004.11147.pdf)
3. **Binary graph neural networks** [CVPR 2021] [[paper]](https://arxiv.org/pdf/2012.15823.pdf)[[code]](https://github.com/mbahri/binary_gnn)
4. **Meta-aggregator: Learning to aggregate for 1-bit graph neural networks** [ICCV 2021] [[paper]](https://arxiv.org/pdf/2109.12872v1.pdf)
5. **BitGNN: Unleashing the Performance Potential of Binary Graph Neural Networks on GPUs** [ICS 2023] [[paper]](https://arxiv.org/pdf/2305.02522.pdf)

<a name="Graph_Condensation" />

## 7. Graph Condensation

1. **Graph Condensation for Graph Neural Networks** [ICLR 2021] [[paper]](https://arxiv.org/pdf/2110.07580.pdf)[[code]](https://github.com/ChandlerBang/GCond)
2. **Condensing graphs via one-step gradient matching** [KDD 2022] [[paper]](https://arxiv.org/pdf/2206.07746.pdf)[[code]](https://github.com/amazon-science/doscond)
3. **Graph condensation via receptive field distribution matching** [Arxiv 2022] [[paper]](https://arxiv.org/pdf/2206.13697.pdf)
4. **Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data** [Arxiv 2023] [[paper]](https://arxiv.org/pdf/2306.02664.pdf)[[code]](https://github.com/Amanda-Zheng/SFGC)
5. **Graph Condensation via Eigenbasis Matching** [Arxiv 2023] [[paper]](https://arxiv.org/pdf/2310.09202.pdf)
6. **Kernel Ridge Regression-Based Graph Dataset Distillation** [KDD 2023] [[paper]](https://dl.acm.org/doi/10.1145/3580305.3599398)[[code]](https://github.com/pricexu/KIDD)
7. **Graph Condensation for Inductive Node Representation Learning** [ICDE 2024] [[paper]](https://arxiv.org/pdf/2307.15967.pdf)
8. **Fast Graph Condensation with Structure-based Neural Tangent Kernel** [Arxiv 2023] [[paper]](https://arxiv.org/pdf/2310.11046.pdf)