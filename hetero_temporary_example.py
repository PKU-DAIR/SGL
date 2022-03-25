import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset.ogbn_mag import OgbnMag
from models.graph_operator import LaplacianGraphOp
from models.simple_models import MultiLayerPerceptron
from simple_models import OneDimConvolution


def mini_batch_evaluate(model, dataset, x_list_list, labels, val_loader, test_loader, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_input = []
        for x_list in x_list_list:
            val_input.append([])
            for x in x_list:
                val_input[-1].append(x[batch].to(device))
        val_output = model(val_input)

        pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += pred.eq(labels[batch]).double().sum()
    for batch in test_loader:
        test_input = []
        for x_list in x_list_list:
            test_input.append([])
            for x in x_list:
                test_input[-1].append(x[batch].to(device))
        test_output = model(test_input)

        pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += pred.eq(labels[batch]).double().sum()

    return correct_num_val / len(dataset.val_idx), correct_num_test / len(dataset.test_idx)


def mini_batch_train(model, dataset, x_list_list, labels, train_loader, loss_fn, device):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        # train_output = model.model_forward(batch, device)
        train_input = []
        for x_list in x_list_list:
            train_input.append([])
            for x in x_list:
                train_input[-1].append(x[batch].to(device))
        train_output = model(train_input)
        loss_train = loss_fn(train_output, labels[batch])

        pred = train_output.max(1)[1].type_as(labels)
        correct_num += pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    return loss_train_sum / len(train_loader), correct_num / len(dataset.train_idx)


class NARS(nn.Module):
    def __init__(self, num_subgraphs, prop_steps, feat_dim, num_classes, hidden_dim, num_layers):
        super(NARS, self).__init__()
        self.__aggregator = OneDimConvolution(num_subgraphs, prop_steps, feat_dim)
        self.__fcs = nn.ModuleList()
        for _ in range(prop_steps):
            self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        self.__out_model = MultiLayerPerceptron(hidden_dim * prop_steps, hidden_dim, num_layers, num_classes)

    def forward(self, feat_list_list):
        aggregated_feat_list = self.__aggregator(feat_list_list)

        concat_feat = self.__fcs[0](aggregated_feat_list[0])
        for i in range(1, len(aggregated_feat_list)):
            feat_temp = F.relu(self.__fcs[i](aggregated_feat_list[i]))
            concat_feat = torch.hstack((concat_feat, feat_temp))

        output = self.__out_model(concat_feat)
        return output


dataset = OgbnMag("mag", "../dataset/")
random_edge_types = list(np.random.choice(dataset.edge_types, size=3, replace=False))
print(random_edge_types)
subgraph_dict = dataset.nars_preprocess(random_edge_types, random_num=2)
predict_class = 'paper'
predict_idx = dataset.data.node_id_dict['paper']

train_loader = torch.utils.data.DataLoader(
    dataset.train_idx, batch_size=10000, shuffle=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(
    dataset.val_idx, batch_size=10000, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(
    dataset.test_idx, batch_size=10000, shuffle=False, drop_last=False)

# subgraph = adj, feature, node_id
graph_op = LaplacianGraphOp(prop_steps=2, r=0.5)

propagated_x_list_list = []
for _ in range(len(subgraph_dict)):
    propagated_x_list_list.append([])
for key in subgraph_dict.keys():
    edge_type_list = []
    for edge_type in key:
        edge_type_list.append(edge_type.split("__")[0])
        edge_type_list.append(edge_type.split("__")[2])
    if predict_class in edge_type_list:
        adj, feature, node_id = subgraph_dict[key]
        propagated_feature = graph_op.propagate(adj, feature)

        start_pos = list(node_id).index(predict_idx[0])
        for i, feature in enumerate(propagated_feature):
            propagated_x_list_list[i].append(feature[start_pos:start_pos + dataset.data.num_node[predict_class]])

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
model = NARS(len(propagated_x_list_list[0]), 2, dataset.data.num_features[predict_class],
             dataset.data.num_classes[predict_class], 256, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

labels = dataset.data[predict_class].y.squeeze(-1).to(device)

t_total = time.time()
best_val = 0.
best_test = 0.
for epoch in range(200):
    t = time.time()
    loss_train, acc_train = mini_batch_train(model, dataset, propagated_x_list_list, labels, train_loader,
                                             nn.CrossEntropyLoss(), device)
    acc_val, acc_test = mini_batch_evaluate(model, dataset, propagated_x_list_list, labels, val_loader, test_loader,
                                            device)
    print('Epoch: {:03d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train),
          'acc_train: {:.4f}'.format(acc_train),
          'acc_val: {:.4f}'.format(acc_val),
          'acc_test: {:.4f}'.format(acc_test),
          'time: {:.4f}s'.format(time.time() - t))
    if acc_val > best_val:
        best_val = acc_val
        best_test = acc_test

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
