import copy
import torch
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g
    
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_scores_gen_graphs(edges_pos, edges_neg, adj_pred, adj_label):
    # get logists and labels
    preds = adj_pred[edges_pos.T]
    preds_neg = adj_pred[edges_neg]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    roc_auc = roc_auc_score(labels, logists)
    ap_score = average_precision_score(labels, logists)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    warnings.simplefilter("ignore", RuntimeWarning)
    f1s = np.nan_to_num(2 * precisions * recalls / (precisions + recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]
    # calc reconstracted adj_mat and accuracy with the threshold for best f1
    adj_rec = copy.deepcopy(adj_pred)
    adj_rec[adj_rec < thresh] = 0
    adj_rec[adj_rec >= thresh] = 1
    labels_all = adj_label.view(-1).long().cpu()
    preds_all = adj_rec.view(-1).long()
    recon_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    results = {"roc": roc_auc, "pr": pr_auc, "ap": ap_score, "pre": pre, "rec": rec, "f1": f1, "acc": recon_acc, "adj_recon": adj_rec}
    
    return results