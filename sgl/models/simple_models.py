import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps, feat_dim):
        super(OneDimConvolution, self).__init__()
        self.__hop_num = prop_steps

        self.__learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            self.__learnable_weight.append(nn.Parameter(
                torch.FloatTensor(feat_dim, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.__learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.__hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (
                    adopted_feat * (self.__learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class OneDimConvolutionWeightSharedAcrossFeatures(nn.Module):
    def __init__(self, num_subgraphs, prop_steps):
        super(OneDimConvolutionWeightSharedAcrossFeatures, self).__init__()
        self.__hop_num = prop_steps

        self.__learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            # To help xvarient_uniform_ calculate fan in and fan out, "1" should be kept here.
            self.__learnable_weight.append(nn.Parameter(
                torch.FloatTensor(1, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.__learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.__hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (
                    adopted_feat * (self.__learnable_weight[i])).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class FastOneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps):
        super(FastOneDimConvolution, self).__init__()

        self.__num_subgraphs = num_subgraphs
        self.__prop_steps = prop_steps

        # How to initialize the weight is extremely important.
        # Pure xavier will lead to extremely unstable accuracy.
        # Initialized with ones will not perform as good as this one.        
        self.__learnable_weight = nn.Parameter(
            torch.ones(num_subgraphs * prop_steps, 1))

    # feat_list_list: 3-d tensor (num_node, feat_dim, num_subgraphs * prop_steps)
    def forward(self, feat_list_list):
        return (feat_list_list @ self.__learnable_weight).squeeze(dim=2)

    @property
    def subgraph_weight(self):
        return self.__learnable_weight.view(
            self.__num_subgraphs, self.__prop_steps).sum(dim=1)

class IdenticalMapping(nn.Module):
    def __init__(self) -> None:
        super(IdenticalMapping, self).__init__()

    def forward(self, feature):
        return feature
        
class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.__fc = nn.Linear(feat_dim, output_dim)

    def forward(self, feature):
        output = self.__fc(feature)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.5, bn=False):
        super(MultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.__num_layers = num_layers

        self.__fcs = nn.ModuleList()
        self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.__fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.__fcs.append(nn.Linear(hidden_dim, output_dim))

        self.__bn = bn
        if self.__bn is True:
            self.__bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.__bns.append(nn.BatchNorm1d(hidden_dim))

        self.__dropout = nn.Dropout(dropout)
        self.__prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.__fcs:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):
        for i in range(self.__num_layers - 1):
            feature = self.__fcs[i](feature)
            if self.__bn is True:
                feature = self.__bns[i](feature)
            feature = self.__prelu(feature)
            feature = self.__dropout(feature)

        output = self.__fcs[-1](feature)
        return output

class ResMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False):
        super(ResMultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError("ResMLP must have at least two layers!")
        self.__num_layers = num_layers

        self.__fcs = nn.ModuleList()
        self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.__fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.__fcs.append(nn.Linear(hidden_dim, output_dim))

        self.__bn = bn
        if self.__bn is True:
            self.__bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.__bns.append(nn.BatchNorm1d(hidden_dim))

        self.__dropout = nn.Dropout(dropout)
        self.__relu = nn.ReLU()

    def forward(self, feature):
        feature = self.__dropout(feature)
        feature = self.__fcs[0](feature)
        if self.__bn is True:
            feature = self.__bns[0](feature)
        feature = self.__relu(feature)
        residual = feature

        for i in range(1, self.__num_layers - 1):
            feature = self.__dropout(feature)
            feature = self.__fcs[i](feature)
            if self.__bn is True:
                feature = self.__bns[i](feature)
            feature_ = self.__relu(feature)
            feature = feature_ + residual
            residual = feature_

        feature = self.__dropout(feature)
        output = self.__fcs[-1](feature)
        return output

class GCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)   
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class SAGEConv(nn.Module):
    """
    Simple GraphSAGE layer, use mean as aggregation way
    """

    def __init__(self, in_features, out_features, normalize=True):
        super(SAGEConv, self).__init__()
        if isinstance(in_features, int):
            in_features = (in_features, in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize

        self.lin_l = nn.Linear(in_features[0], out_features)
        self.lin_r = nn.Linear(in_features[1], out_features)
        if normalize:
            self.norm = lambda x: F.normalize(x, p=1, dim=1)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, adj):
        output = torch.spmm(adj, x)
        output = self.lin_l(output)

        num_tgt = adj.shape[0]
        x_r = x[:num_tgt]
        output += self.lin_r(x_r)

        if self.normalize:
            output = self.norm(output)
        
        return output
    
class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5):
        super(SAGE, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(SAGEConv(nfeat, nhid))
        self.nlayers = nlayers
        for _ in range(nlayers-2):
            self.gcs.append(SAGEConv(nhid, nhid))
        self.gcs.append(SAGEConv(nhid, nclass, normalize=False))
        self.dropout = dropout

    def reset_parameter(self):
        for conv in self.gcs:
            conv.reset_parameters()

    def forward(self, x, block):
        repr = x
        if len(block) == self.nlayers:
            for i in range(self.nlayers-1):
                repr = self.gcs[i](repr, block[i])
                repr = F.relu(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[-1])
        elif len(block) == 1:
            for gc in self.gcs[:-1]:
                repr = gc(repr, block[0])
                repr = F.relu(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[0])
        else:
            raise ValueError('The sampling layer must be equal to GNN layer.')
        
        return F.log_softmax(repr, dim=1)
    
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.gcs):
            xs = []
            for batch in subgraph_loader:
                batch_in, _, block = batch
                block.to_device(device)
                x = x_all[batch_in].to(device)
                x = conv(x, block[0]) # one-layer sampling
                if i != self.nlayers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, layer=GCNConv, nlayers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(layer(nfeat, nhid))
        self.nlayers = nlayers
        for _ in range(nlayers-2):
            self.gcs.append(layer(nhid, nhid))
        self.gcs.append(layer(nhid, nclass))
        self.dropout = dropout
    
    def reset_parameter(self):
        for conv in self.gcs:
            conv.reset_parameters()

    def forward(self, x, block):
        repr = x
        if len(block) == self.nlayers:
            for i in range(self.nlayers-1):
                repr = self.gcs[i](repr, block[i])
                repr = F.relu(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[-1])
        elif len(block) == 1:
            for gc in self.gcs[:-1]:
                repr = gc(repr, block[0])
                repr = F.relu(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[0])
        else:
            raise ValueError('The sampling layer must be equal to GNN layer.')
        
        return F.log_softmax(repr, dim=1)
    
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.gcs):
            xs = []
            for batch in subgraph_loader:
                batch_in, _, block = batch
                block.to_device(device)
                x = x_all[batch_in].to(device)
                x = conv(x, block[0]) # one-layer sampling
                if i != self.nlayers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all