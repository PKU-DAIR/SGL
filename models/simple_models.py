import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps, feat_dim):
        super(OneDimConvolution, self).__init__()
        self.__hop_num = prop_steps

        self.__learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            self.__learnable_weight.append(nn.Parameter(torch.FloatTensor(feat_dim, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.__learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.__hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (adopted_feat * (self.__learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list
    

class OneDimConvolutionWeightSharedAcrossFeatures(nn.Module):
    def __init__(self, num_subgraphs, prop_steps, feat_dim):
        super(OneDimConvolutionWeightSharedAcrossFeatures, self).__init__()
        self.__hop_num = prop_steps
        self.__feat_dim = feat_dim

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


class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.__fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feature):
        output = self.__fc(feature)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bn=False):
        super(MultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.__num_layers = num_layers

        self.__fcs = nn.ModuleList()
        self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.__fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.__fcs.append(nn.Linear(hidden_dim, num_classes))

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
