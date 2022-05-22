import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.nn.functional as F
import scipy.sparse as sp

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import set_seed, edge_predict_eval, mask_test_edges, edge_predict_train, \
    mini_batch_edge_predict_train, mini_batch_edge_predict_eval, edge_predict_score, mix_pos_neg_edges
from sgl.tasks.utils import sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm

class LinkPredictionGAE(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, loss_fn=F.binary_cross_entropy_with_logits, seed=42,
                 train_batch_size=None, eval_batch_size=None, threshold=0.5):
        super(LinkPredictionGAE, self).__init__()

        self.__dataset = dataset
        self.__edge_labels = self._generate_edge_labels() # not used yet, maybe in the future
        
        self.__train_adj, self.__train_edges, self.__train_edges_neg, self.__val_edges, self.__val_edges_neg, self.__test_edges, self.__test_edges_neg = \
            mask_test_edges(self.__dataset.adj)
        print("Edge split finished!")
        
        self.__all_edges = torch.cat((self.__train_edges, self.__val_edges, self.__test_edges))
        self.__all_edges_neg = torch.cat((self.__train_edges_neg, self.__val_edges_neg, self.__test_edges_neg))

        self.__model = model
        self.__with_params = len(list(model.parameters())) != 0 
        self.__optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if self.__with_params is True else None
        
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed
        self.__pred_threshold = threshold

        self.__mini_batch = False
        if train_batch_size is not None:
            self.__mini_batch = True

            train_all_edges, train_all_lables = mix_pos_neg_edges(self.__train_edges, self.__train_edges_neg, train_batch_size // 2)
            train_dataset = Data.TensorDataset(train_all_edges, train_all_lables)

            val_all_edges, val_all_lables = mix_pos_neg_edges(self.__val_edges, self.__val_edges_neg, eval_batch_size // 2)
            val_dataset = Data.TensorDataset(val_all_edges, val_all_lables)

            test_all_edges, test_all_lables = mix_pos_neg_edges(self.__test_edges, self.__test_edges_neg, eval_batch_size // 2)
            test_dataset = Data.TensorDataset(test_all_edges, test_all_lables)

            all_edges, all_edges_label = mix_pos_neg_edges(self.__all_edges, self.__all_edges_neg, eval_batch_size // 2)
            all_dataset = Data.TensorDataset(all_edges, all_edges_label)

            self.__train_loader = DataLoader(
                dataset=train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.__val_loader = DataLoader(
                dataset=val_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.__test_loader = DataLoader(
                dataset=test_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)

            # For post_process evaluation, now is not used
            self.__all_eval_loader = DataLoader(
                dataset=all_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)

        self.__test_roc_auc, self.__test_avg_prec = self._execute()

    @property
    def test_roc_auc(self):
        return self.__test_roc_auc

    @property
    def test_avg_prec(self):
        return self.__test_avg_prec

    def _generate_edge_labels(self):
        adj = (self.__dataset.adj + sp.eye(self.__dataset.num_node)).toarray()
        adj_label = torch.FloatTensor(adj)
        return adj_label

    def _execute(self):
        set_seed(self.__seed)

        pre_time_st = time.time()
        self.__model.preprocess(self.__train_adj, self.__dataset.x)
        pre_time_ed = time.time()
        print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")

        self.__model = self.__model.to(self.__device)
        self.__edge_labels = self.__edge_labels.to(self.__device)

        train_node_index = torch.arange(0, self.__dataset.num_node, dtype=torch.long)

        t_total = time.time()
        best_roc_auc_val, best_avg_prec_val = 0. , 0.
        best_roc_auc_test, best_avg_prec_test = 0. , 0.

        for epoch in range(self.__epochs):
            t = time.time()
            if self.__mini_batch is False:
                loss_train, roc_auc_train, avg_prec_train = edge_predict_train(self.__model, train_node_index, self.__with_params, self.__train_edges, self.__train_edges_neg,
                                                                               self.__device, self.__optimizer, self.__loss_fn, self.__pred_threshold)
                roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test = edge_predict_eval(self.__model, train_node_index, self.__val_edges, self.__val_edges_neg,
                                                                                           self.__test_edges, self.__test_edges_neg, self.__device, self.__pred_threshold)
            else:
                loss_train, roc_auc_train, avg_prec_train = mini_batch_edge_predict_train(self.__model, train_node_index, self.__with_params, self.__train_loader, self.__device, 
                                                                                    self.__optimizer, self.__loss_fn, self.__pred_threshold)
                roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test = mini_batch_edge_predict_eval(self.__model, train_node_index, self.__val_loader, self.__test_loader, self.__device, self.__pred_threshold)
            print('Epoch: {:03d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'roc_auc_train: {:.4f}'.format(roc_auc_train),
                  'avg_prec_train: {:.4f}'.format(avg_prec_train),
                  '\n',
                  'roc_auc_val: {:.4f}'.format(roc_auc_val),
                  'avg_prec_val: {:.4f}'.format(avg_prec_val),
                  'roc_auc_test: {:.4f}'.format(roc_auc_test),
                  'avg_prec_test: {:.4f}'.format(avg_prec_test),
                  'time: {:.4f}s'.format(time.time() - t))

            if roc_auc_val > best_roc_auc_val:
                best_roc_auc_val = roc_auc_val
                best_roc_auc_test = roc_auc_test
            
            if avg_prec_val > best_avg_prec_val:
                best_avg_prec_val = avg_prec_val
                best_avg_prec_test = avg_prec_test

        roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test = self._postprocess()
        if roc_auc_val > best_roc_auc_val:
            best_roc_auc_val = roc_auc_val
            best_roc_auc_test = roc_auc_test
        if avg_prec_val > best_avg_prec_val:
            best_avg_prec_val = avg_prec_val
            best_avg_prec_test = avg_prec_test

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best roc_auc_val: {best_roc_auc_val:.4f}, best roc_auc_test: {best_roc_auc_test:.4f}')
        print(f"Best avg_prec_val: {best_avg_prec_val:.4f}, best avg_prec_test: {best_avg_prec_test:.4f}")
        return best_roc_auc_test, best_avg_prec_test

    def _postprocess(self):
        self.__model.eval()
        if self.__mini_batch is False:
            node_features = self.__model.model_forward(
                range(self.__dataset.num_node), self.__device).to('cpu')
        else:
            # postprocess does not support batch training yet
            outputs = None
            return 0., 0., 0., 0.

        final_node_features = self.__model.postprocess(self.__train_adj, node_features)
        edge_feature = torch.mm(final_node_features, final_node_features.t()).data

        roc_auc_val, avg_prec_val = edge_predict_score(edge_feature, self.__val_edges, self.__val_edges_neg, self.__pred_threshold)
        roc_auc_test, avg_prec_test = edge_predict_score(edge_feature, self.__test_edges, self.__all_edges_neg, self.__pred_threshold)
        return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test

class LinkPredictionNAFS(BaseTask):
    """ LinkPrediction Tasks based on NAFS
        Fast and Does Not need train
    """
    def __init__(self, dataset, hops=20, method='mean', seed=42, r_list=[0.5, 0.4, 0.3, 0.2, 0.1, 0], threshold=0.5):
        # NOTE that: hops can be either an integer or a list
        # when it's a list, it contains the numbers of hops that you want to test
        # when it's a numebr, it will test all the hops in range(hops) 
        method = method.lower()
        if method not in ['mean', 'max', 'concat', 'simple']:
            raise ValueError("Method not Suppoted! Choose 'mean', 'max' or 'concat' !")  
        if type(hops) not in [list, int]:
            raise ValueError("hops type not supported!")

        super(LinkPredictionNAFS, self).__init__()
        self.__dataset = dataset

        self.__train_adj, _, _, _, _, self.__test_edges, self.__test_edges_neg = mask_test_edges(self.__dataset.adj)
        print("Edge split finished!")

        self.__method = method
        self.__r_list = r_list
        self.__hops = range(hops) if type(hops) == int else hops
        self.__seed = seed
        self.__pred_threshold = threshold

        self.__best_hop_roc_auc, self.__best_hop_avg_prec, self.__test_roc_auc, self.__test_avg_prec = self._execute()
    
    @property
    def test_roc_auc(self):
        return self.__test_roc_auc

    @property
    def test_avg_prec(self):
        return self.__test_avg_prec

    @property
    def best_hop_roc_auc(self):
        return self.__best_hop_roc_auc

    @property
    def best_hop_avg_prec(self):
        return self.__best_hop_avg_prec

    def _execute(self):
        set_seed(self.__seed)

        t_total = time.time()
        best_roc_auc, best_avg_prec = 0., 0. 
        best_hop_roc_auc, best_hop_avg_prev = 0, 0

        for hop in self.__hops:
            t = time.time()
            roc_auc, avg_prec = self._k_hop_link_prediction(hop)

            print('hops:{:2d}'.format(hop),
                  'roc_auc_score: {:.4f}'.format(roc_auc),
                  'avg_precision: {:.4f}'.format(avg_prec),
                  'time: {:.4f} seconds'.format(time.time() - t)
                  )
            
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_hop_roc_auc = hop
            if avg_prec > best_avg_prec:
                best_avg_prec = avg_prec
                best_hop_avg_prev = hop
            
        print("Node Smoothing Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("best_roc_auc_score: {:.4f}, best_avg_precision: {:.4f}".format(best_roc_auc, best_avg_prec))

        return best_hop_roc_auc, best_hop_avg_prev, best_roc_auc, best_avg_prec
        
    def _k_hop_link_prediction(self, hops):
        input_features = []
        features = torch.tensor(self.__dataset.x, dtype=torch.float)
        r_list = self.__r_list
        for r in r_list:
            t = time.time()
            adj_norm = adj_to_symmetric_norm(self.__train_adj, r)
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
            features_list = []
            features_list.append(features)
            for _ in range(hops):
                features_list.append(torch.spmm(adj_norm, features_list[-1]))

            if self.__method == 'simple':
                input_features.append(features_list[-1])
                break

            weight_list = []
            norm_fea = torch.norm(features, 2, 1).add(1e-10)
            for fea in features_list:
                norm_cur = torch.norm(fea, 2, 1).add(1e-10)
                tmp = torch.div((features * fea).sum(1), norm_cur)
                tmp = torch.div(tmp, norm_fea)

                weight_list.append(tmp.unsqueeze(-1))

            tmp_weight = torch.cat(weight_list, dim=1)
            weight = F.softmax(tmp_weight, dim=1)

            input_feas = []
            for i in range(self.__dataset.num_node):
                fea = 0.
                for j in range(hops + 1):
                    fea += (weight[i][j]*features_list[j][i]).unsqueeze(0)
                input_feas.append(fea)
            input_feas = torch.cat(input_feas, dim=0)
            input_features.append(input_feas)
            print("hops {:d}: r={:.2f} calculation finished in {:.4f} seconds".format(hops, r, time.time() - t))
        
        if self.__method == 'mean':
            input_features = sum(input_features) / len(input_features)
        elif self.__method == 'max':
            input_features =  list(map(lambda x: x.unsqueeze(0), input_features))
            input_features = torch.cat(input_features, dim=0).max(0)[0]
        elif self.__method == 'concat':
            input_features = torch.cat(input_features, dim=1)
        elif self.__method == 'simple':
            input_features = input_features[-1]

        sim = torch.mm(input_features, input_features.t())
        roc_auc, avg_prec = edge_predict_score(sim, self.__test_edges, self.__test_edges_neg, self.__pred_threshold)
        return roc_auc, avg_prec
