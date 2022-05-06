# Tricks of GNNs

## Simple Graph Augmentation

- **Label Usage**
  
  <img src="img/Label_Reuse.jpg" alt="Label_Reuse" style="zoom: 80%;" />
  
  - **Label Input**
    - In each epoch, randomly split the training set into two sub-datasets ![formula](https://render.githubusercontent.com/render/math?math=D_%7B%5Crm%20train%7D%5EL) and ![formula](https://render.githubusercontent.com/render/math?math=D_%7B%5Crm%20train%7D%5EU)
    - For nodes in ![formula](https://render.githubusercontent.com/render/math?math=D_%7B%5Crm%20train%7D%5EL) , concatenate their label information with the original features
  - **Label Reuse**
    - Similar to label propagation algorithm. In each iteration, assign the output soft labels to the nodes which didn’t have true labels
    - Enable the model to update the predicted labels through iterations, but not limited by the homophily assumption
  - **Optimized Label Reuse**
    - The reliability of the output soft labels is not guaranteed in the first few epochs, therefore **Label Reuse** may not be effective in these occasions 
    - Optimal optimization includes setting threshold, only exploiting the output soft labels at later stages in training, etc.
  - [Ref_code](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)

- **Label embedding and propagation**
  
  - Used in node prediction task. Get matrix ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20Y) through label propagation on training set labels. Apply linear transformation to ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20Y) and add the result to the original features to create the new features
    
    ![formula](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0A%20%20%20%20%5Chat%20Y%20%26%3D%20A%5Ek%20Y_%7B%5Crm%20train%7D%5C%5C%0A%20%20%20%20X_%7B%5Crm%20new%7D%26%3DX%2BW%5Chat%20Y%0A%20%20%20%20%5Cend%7Baligned%7D)
  
  - [Paddle_code](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp)

- **Virtual Nodes**
  
  - Used in graph classification task. Add several virtual nodes ( such as 3 ) to the original training set, and all nodes are connected to these virtual nodes. Their features are the average of the remaining node features.
  - [Paddle_code](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/ogbn-arxiv/unimp_appnp_vnode_smooth)

## Model Architecture Design

- **Transition matrix normalization for GAT**
  
  - Normalize the transition matrix of GAT
    
    ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20A_%7B%5Crm%20attention-norm%7D%3D%5Chat%20D%5E%7B-1%2F2%7D%5Chat%20A_%7B%5Crm%20attention%7D%5Chat%20D%5E%7B-1%2F2%7D)
  
  - [Ref_code](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)

- **Correct and Smooth**
  
  - Smoothen errors in **Correct** steps and smoothen labels in **Smooth** steps
  - [C&S_code](https://github.com/CUAI/CorrectAndSmooth)

- **Hop-wise attention (AGDN)**
  
  - In each layer of the model, the information of different scales is calculated by ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20H%5E%7B%28k%2Cl%29%7D%3D%5Chat%20T%5Ek%20H%5El), where ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20T) is the transformation matrix of GAT
  
  - When the attention weight of each node is calculated on the information of different scales, ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20H_i%5E%7B%280%2Cl%29%7D) is used as benchmark: 
    
    <img src='./img/AGDN_formula.jpg'  alt="AGDN_formula" style="zoom: 80%;" />
  
  - [AGDN_code](https://github.com/skepsun/adaptive_graph_diffusion_networks_with_hop-wise_attention)

- **Residual term**
  
  - Add residual terms to each layer of the model
    
    ![formula](https://render.githubusercontent.com/render/math?math=X%5E%7B%28k%2B1%29%7D%3D%5Csigma%5Cleft%28%5Chat%20TX%5E%7B%28k%29%7DW_0%5E%7B%28k%29%7D%2BX%5E%7B%28k%29%7DW_1%5E%7B%28k%29%7D%5Cright%29)
    
    where ![formula](https://render.githubusercontent.com/render/math?math=%5Chat%20T) is the transformation of GCN or GAT
  
  - [Ref Code](https://github.com/skepsun/adaptive_graph_diffusion_networks_with_hop-wise_attention)

- **Self Knowledge Distillation**
  
  - Firstly train a pretrained teacher model, and then use logits of the teacher model and ground truth labels to train a student model with KD loss and CE loss
  - The architecture of the student model is the same as teacher model
  - [Self-KD-code](https://github.com/ShunliRen/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)

- **APPNP-like Propagation (UniMP)** 
  
  - Add the node features and labels vectors together as propagation information ![formula](https://render.githubusercontent.com/render/math?math=%28H%5E0%3DX%2B%5Chat%20Y_d%29)
  
  - Apply APPNP-like propagation to the combined message, where the transformation matrix can be attention matrix of GAT or Graph Transformer

      ![formula](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%20%20%20%20H%5E%7B%280%29%7D%26%3DX%2B%5Chat%7BY%7DW%5E%7B%28l%29%7D%5C%5C%0A%20%20%20%20H%5E%7B%28k%29%7D%26%3D%5Csigma%28%28%281-%5Calpha%29A%5E%7B%2A%7D%2B%5Calpha%20I%29H%5E%7B%28k-1%29%7DW%5E%7B%28k%29%7D%29%0A%5Cend%7Bsplit%7D)
  
  - [UniMP_code](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp)

## Training Tricks

- **Masked label prediction** 
  
  - In each epoch, randomly mask labels of some nodes in training set and only make use of the label information of the unmasked nodes for training
  - This trick can be combined with tricks in **Simple Graph Augmentation** part
  - [Paddle_code](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp)

- **FLAG**
  
  - Add a gradient-based perturbation to the feature during training to achieve data enhancement and alleviate overfit
    
    <img src="img/FLAG_image.jpg" alt="FLAG_img" style="zoom: 80%;" />
  
  - [FLAG_code](https://github.com/devnkong/FLAG)

## Loss Function Modification

- **A More Robust Cross Entropy Loss**
  
  - Replace the binary cross entropy loss with LCE loss, which can be written as 
    
    ![formula](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0A%20%20%20%20%5Cmathcal%7BL%7D_%7B%5Ctext%7BLBCE%7D%7D%28y%2C%5Cwidetilde%7By%7D%29%26%3D%0A%20%20%20%20%5Clog%28%5Cmathcal%7BL%7D_%7B%5Ctext%7BBCE%7D%7D%28y%2C%5Cwidetilde%7By%7D%29%2B%5Cepsilon%29%5C%5C%0A%20%20%20%20%26%3D%5Clog%28%5Cepsilon-y%5Clog%28%5Cwidetilde%7By%7D%29-%0A%20%20%20%20%281-y%29%5Clog%281-%5Cwidetilde%7By%7D%29%29%0A%20%20%20%20%5Cend%7Baligned%7D)
    
    where ![formula](https://render.githubusercontent.com/render/math?math=%5Cepsilon%20%3E%200) is a hyperparameter.
  
  - Comparison with BCE loss and LBCE loss, where ![formula](https://render.githubusercontent.com/render/math?math=y%3D1%2C%5Cepsilon%3D1-%5Clog2)
    
    <img src="img/LBCE_loss.png" alt="LBCE_loss" style="zoom:67%;" />
  
  - [Ref_code](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)

- **Topology Loss**
  
  - Randomly sample a portion of edges and maximize the sum of cosine similarity of the two endpoint embeddings of the edges
    
    ![formula](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D_%7B%5Crm%20topo%7D%3D%5Cfrac%7B1%7D%7B%7CE%27%7C%7D%5Csum_%7Be%5Cin%20E%27%7D%5Ccos%5Cleft%28p_%7Be%5Es%7D%2Cp_%7Be%5Ed%7D%5Cright%29)
  
  - [Topo Loss code](https://github.com/mengyangniu/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)
