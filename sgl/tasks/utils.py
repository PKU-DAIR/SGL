import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score

from sgl.tasks.clustering_metrics import clustering_metrics

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def add_labels(features, labels, idx, num_classes):
    onehot = np.zeros([features.shape[0], num_classes])
    onehot[idx, labels[idx]] = 1
    return np.concatenate([features, onehot], axis=-1)

def evaluate(model, val_idx, test_idx, labels, device):
    model.eval()
    val_output = model.model_forward(val_idx, device)
    test_output = model.model_forward(test_idx, device)

    acc_val = accuracy(val_output, labels[val_idx])
    acc_test = accuracy(test_output, labels[test_idx])
    return acc_val, acc_test


def mini_batch_evaluate(model, val_idx, val_loader, test_idx, test_loader, labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_output = model.model_forward(batch, device)
        pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += pred.eq(labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_idx)

    for batch in test_loader:
        test_output = model.model_forward(batch, device)
        pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += pred.eq(labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_idx)

    return acc_val.item(), acc_test.item()


def train(model, train_idx, labels, device, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    train_output = model.model_forward(train_idx, device)
    loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train


def mini_batch_train(model, train_idx, train_loader, labels, device, optimizer, loss_fn):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        train_output = model.model_forward(batch, device)
        loss_train = loss_fn(train_output, labels[batch])

        pred = train_output.max(1)[1].type_as(labels)
        correct_num += pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_idx)

    return loss_train, acc_train.item()


def cluster_loss(train_output, y_pred, cluster_centers):

    for i in range(len(cluster_centers)):
        if i == 0:
            dist = torch.norm(train_output - cluster_centers[i], p=2, dim=1, keepdim=True)
        else:
            dist = torch.cat((dist, torch.norm(train_output - cluster_centers[i], p=2, dim=1, keepdim=True)), 1)
    
    loss = 0.
    loss_tmp = -dist.mean(1).sum()
    loss_tmp += 2 * np.sum(dist[j, x] for j, x in zip(range(dist.shape[0]), y_pred))
    loss = loss_tmp / dist.shape[0]
    return loss


def clustering_train(model, train_idx, labels, device, optimizer, loss_fn, n_clusters, n_init):
    model.train()
    optimizer.zero_grad()

    train_output = model.model_forward(train_idx, device)
    
    # calc loss
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    y_pred = kmeans.fit_predict(train_output.data.cpu().numpy()) # cluster_label
    cluster_centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    loss_train = loss_fn(train_output, y_pred, cluster_centers)
    loss_train.backward()
    optimizer.step()

    # calc acc, nmi, adj
    labels = labels.cpu().numpy()
    cm = clustering_metrics(labels, y_pred)
    acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()

    return loss_train.item(), acc, nmi, adjscore


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    edges_all_set = set()
    for edge in edges_all.tolist():
        edges_all_set.add(tuple(edge))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b):
        return a in b

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember( (idx_i, idx_j), edges_all_set):
            continue
        if train_edges_false:
            if ismember( (idx_j, idx_i), train_edges_false):
                continue
            if ismember( (idx_i, idx_j), train_edges_false):
                continue
        train_edges_false.add((idx_i, idx_j))

    test_edges_false = set()
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember( (idx_i, idx_j), edges_all_set):
            continue
        if ismember( (idx_i, idx_j), train_edges_false):
            continue
        if ismember( (idx_j, idx_i), train_edges_false):
            continue
        if test_edges_false:
            if ismember( (idx_j, idx_i), test_edges_false):
                continue
            if ismember( (idx_i, idx_j), test_edges_false):
                continue
        test_edges_false.add((idx_i, idx_j))

    val_edges_false = set()
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember( (idx_i, idx_j), edges_all_set):
            continue
        if ismember( (idx_i, idx_j), train_edges_false):
            continue
        if ismember( (idx_j, idx_i), train_edges_false):
            continue
        if ismember( (idx_i, idx_j), test_edges_false):
            continue
        if ismember( (idx_j, idx_i), test_edges_false):
            continue
        if val_edges_false:
            if ismember( (idx_j, idx_i), np.array(val_edges_false)):
                continue
            if ismember( (idx_i, idx_j), np.array(val_edges_false)):
                continue
        val_edges_false.add((idx_i, idx_j))

    #assert ~ismember(test_edges_false, edges_all)
    #assert ~ismember(val_edges_false, edges_all)
    #assert ~ismember(val_edges, train_edges)
    #assert ~ismember(test_edges, train_edges)
    #assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    train_edges_false = list(train_edges_false)
    val_edges_false = list(val_edges_false)
    test_edges_false = list(test_edges_false)

    train_edges, train_edges_false  = torch.tensor(train_edges), torch.tensor(train_edges_false)
    val_edges, val_edges_false = torch.tensor(val_edges), torch.tensor(val_edges_false)
    test_edges, test_edges_false = torch.tensor(test_edges), torch.tensor(test_edges_false)

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


# input full edge_features, pos_edges and neg_edges to calc roc_auc, avg_prec score
def edge_predict_score(edge_feature, pos_edges, neg_edges, threshold):
    labels = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))))
    all_edges = torch.cat((pos_edges, neg_edges))
    edge_pred = edge_feature[all_edges[:, 0], all_edges[:, 1]].reshape(-1)
    edge_pred = torch.sigmoid(edge_pred)
    # edge_pred = edge_pred > threshold
    roc_auc = roc_auc_score(labels, edge_pred)
    avg_prec = average_precision_score(labels, edge_pred)
    return roc_auc, avg_prec


def edge_predict_train(model, train_node_index, with_params, pos_edges, neg_edges, 
                       device, optimizer, loss_fn, threshold):
    if with_params is True:
        model.train()
        optimizer.zero_grad()

    train_output = model.model_forward(train_node_index, device)
    edge_feature = torch.mm(train_output, train_output.t())
    labels = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges)))).to(device)
    train_edge = torch.cat((pos_edges, neg_edges)).to(device)
    edge_pred = edge_feature[train_edge[:, 0], train_edge[:, 1]].reshape(-1)
    edge_pred = torch.sigmoid(edge_pred)

    # print("-----------------------------")
    # print("edge_features:  ", edge_feature[:200])
    # print("edge_pred:\n", edge_pred[len(pos_edges)-50:len(pos_edges)+50])
    # print("labels:\n",labels[len(pos_edges)-50:len(pos_edges)+50])
    # print("-----------------------------")

    loss = loss_fn(edge_pred, labels)
    if with_params is True:
        loss.backward()
        optimizer.step()

    labels = labels.cpu().data
    edge_pred = edge_pred.cpu().data
    edge_pred = edge_pred > threshold
    roc_auc = roc_auc_score(labels, edge_pred)
    avg_prec = average_precision_score(labels, edge_pred)
    return loss.item(), roc_auc, avg_prec


def edge_predict_eval(model, train_node_index, val_pos_edges, val_neg_edges, 
                      test_pos_edges, test_neg_edges, device, threshold):
    model.eval()
    train_output = model.model_forward(train_node_index, device)
    edge_feature = torch.mm(train_output, train_output.t()).cpu().data

    roc_auc_val, avg_prec_val = edge_predict_score(edge_feature, val_pos_edges, val_neg_edges, threshold)
    roc_auc_test, avg_prec_test = edge_predict_score(edge_feature, test_pos_edges, test_neg_edges, threshold)

    return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test


def mini_batch_edge_predict_train(model, train_node_index, with_params, train_loader, 
                                  device, optimizer, loss_fn, threshold):
    if with_params is True:
        model.train()
        optimizer.zero_grad()
    
    loss_train = 0.
    roc_auc_sum = 0.
    avg_prec_sum = 0.

    output = model.model_forward(train_node_index, device)
    output = output.cpu()
    edge_feature = torch.mm(output, output.t())
    edge_feature = torch.sigmoid(edge_feature)

    for batch, label in train_loader:
        edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
        # print("-----------------------------")
        # print("edge_pred:\n", edge_pred.data[:100])
        # print("labels:\n",label.data[:100])
        # print("roc_auc_partial: ",roc_auc_score(label.data, edge_pred.data[:100]))
        # print("-----------------------------")
        pred_label = edge_pred > threshold
        roc_auc_sum += roc_auc_score(label.data, pred_label.data)
        avg_prec_sum += average_precision_score(label.data, pred_label.data)

        edge_pred = edge_pred.to(device)
        label = label.to(device)
        loss_train += loss_fn(edge_pred, label)

    if with_params is True:
        loss_train.backward()
        optimizer.step()
        
    loss_train = loss_train.item() / len(train_loader)
    roc_auc = roc_auc_sum / len(train_loader)
    avg_prec = avg_prec_sum / len(train_loader)

    return loss_train, roc_auc, avg_prec


def mini_batch_edge_predict_eval(model, train_node_index, val_loader, test_loader, device, threshold):
    model.eval()
    roc_auc_val_sum, avg_prec_val_sum = 0., 0.
    roc_auc_test_sum, avg_prec_test_sum = 0., 0.

    output = model.model_forward(train_node_index, device)
    output = output.cpu().data
    edge_feature = torch.mm(output, output.t())
    edge_feature = torch.sigmoid(edge_feature)

    for batch, label in val_loader:
        edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
        label_pred = edge_pred > threshold
        roc_auc_val_sum += roc_auc_score(label, label_pred)
        avg_prec_val_sum += average_precision_score(label, label_pred)

    roc_auc_val = roc_auc_val_sum / len(val_loader)
    avg_prec_val = avg_prec_val_sum / len(val_loader)

    for batch, label in test_loader:
        edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
        label_pred = edge_pred > threshold
        roc_auc_test_sum += roc_auc_score(label, edge_pred)
        avg_prec_test_sum += average_precision_score(label, edge_pred)

    roc_auc_test = roc_auc_test_sum / len(test_loader)
    avg_prec_test = avg_prec_test_sum / len(test_loader)

    return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test


def mix_pos_neg_edges(pos_edges, neg_edges, mix_size):
    start, end = 0, mix_size
    mix_edges = torch.cat((pos_edges[start:end], neg_edges[start:end]))
    mix_labels = torch.cat((torch.ones(end - start), torch.zeros(end - start)))

    start += mix_size
    end += mix_size
    while end < len(pos_edges):
        tmp_edges = torch.cat((pos_edges[start:end], neg_edges[start:end]))
        tmp_labels = torch.cat((torch.ones(end - start), torch.zeros(end - start)))
        mix_edges = torch.cat((mix_edges, tmp_edges))
        mix_labels = torch.cat((mix_labels, tmp_labels))
        start += mix_size
        end += mix_size
    
    tmp_edges = torch.cat((pos_edges[start:], neg_edges[start:]))
    tmp_labels = torch.cat((torch.ones(len(pos_edges) - start), torch.zeros(len(neg_edges) - start)))
    mix_edges = torch.cat((mix_edges, tmp_edges))
    mix_labels = torch.cat((mix_labels, tmp_labels))

    return mix_edges, mix_labels

def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
