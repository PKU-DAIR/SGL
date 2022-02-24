import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.__fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feature):
        output = self.__fc(feature)
        return F.softmax(output, dim=1)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes, dropout=0.5, residual=False, bn=False):
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

        self.__residual = residual
        self.__dropout = nn.Dropout(dropout)

    def forward(self, feature):
        feature = self.__fcs[0](feature)
        feature = F.relu(feature)
        feature = self.__dropout(feature)
        if self.__bn is True:
            feature = self.__bns[0](feature)

        for i in range(1, self.__num_layers - 1):
            if self.__residual is True:
                feature = self.__fcs[i](feature) + feature
            else:
                feature = self.__fcs[i](feature)
            feature = F.relu(feature)
            feature = self.__dropout(feature)
            if self.__bn is True:
                feature = self.__bns[i](feature)

        output = self.__fcs[-1](feature)
        return F.softmax(output, dim=1)
