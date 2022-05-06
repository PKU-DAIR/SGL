import time
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans

from sgl.tasks.base_task import BaseTask
from sgl.tasks.utils import set_seed, clustering_train, cluster_loss, \
    sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm
from sgl.tasks.clustering_metrics import clustering_metrics

class NodeClustering(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, loss_fn=cluster_loss, seed=42,
                 train_batch_size=None, eval_batch_size=None, n_init=20):
        super(NodeClustering, self).__init__()

        # clustering task does not support batch training
        if train_batch_size is not None or eval_batch_size is not None:
            raise ValueError("clustering task does not support batch training")
        
        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__model = model
        self.__optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.__epochs = epochs
        self.__loss_fn = loss_fn
        self.__device = device
        self.__seed = seed

        # clustering task does not need valid set
        self.__cluster_train_idx = torch.arange(0, self.__dataset.num_node, dtype=torch.long)
        
        # params for Kmeans
        # note that the n_clusters should be equal to the number of different labels
        self.__n_clusters = self.__dataset.num_classes
        self.__n_init = n_init

        self.__acc, self.__nmi, self.__adjscore = self._execute()

    @property
    def acc(self):
        return self.__acc
    
    @property
    def nmi(self):
        return self.__nmi
    
    @property
    def adjscore(self):
        return self.__adjscore

    def _execute(self):
        set_seed(self.__seed)

        pre_time_st = time.time()
        self.__model.preprocess(self.__dataset.adj, self.__dataset.x)
        pre_time_ed = time.time()
        print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")

        self.__model = self.__model.to(self.__device)
        self.__labels = self.__labels.to(self.__device)

        t_total = time.time()
        best_acc = 0.
        best_nmi = 0.
        best_adjscore = 0.

        for epoch in range(self.__epochs):
            t = time.time()

            loss_train, acc, nmi, adjscore = clustering_train(self.__model, self.__cluster_train_idx, self.__labels,
            self.__device, self.__optimizer, self.__loss_fn, self.__n_clusters, self.__n_init)
                
            print("Epoch: {:03d}".format(epoch + 1),
                  "loss_train: {:.4f}".format(loss_train),
                  "acc: {:.4f}".format(acc),
                  "nmi: {:.4f}".format(nmi),
                  "adjscore: {:.4f}".format(adjscore),
                  "time: {:.4f}s".format(time.time() - t)
                )
            
            if acc > best_acc:
                best_acc = acc
            if nmi > best_nmi:
                best_nmi = nmi
            if adjscore > best_adjscore:
                best_adjscore = adjscore

        # postprocess
        acc, nmi, adjscore = self._postprocess()
        if acc > best_acc:
            best_acc = acc
        if nmi > best_nmi:
            best_nmi = nmi
        if adjscore > best_adjscore:
            best_adjscore = adjscore

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f"Best acc: {best_acc:.4f}, best_nmi: {best_nmi:.4f}, best_adjscore: {best_adjscore:.4f}")
        
        return best_acc, best_nmi, best_adjscore

    def _postprocess(self):
        self.__model.eval()
        outputs = self.__model.model_forward(
            range(self.__dataset.num_node), self.__device).to("cpu")
        
        final_output = self.__model.postprocess(outputs)
        kmeans = KMeans(n_clusters=self.__n_clusters, n_init=self.__n_init)
        y_pred = kmeans.fit_predict(final_output.data.cpu().numpy()) # cluster_label
        
        labels = self.__labels.cpu().numpy()
        cm = clustering_metrics(labels, y_pred)
        acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()

        return acc, nmi, adjscore
        
class NodeClusteringNAFS(BaseTask):
    """ NodeClustering Tasks based on NAFS
        Fast and Does Not need train
    """
    def __init__(self, dataset, hops=20, method='mean', seed=42, n_init=20, r_list=[0.5, 0.4, 0.3, 0.2, 0.1, 0]):
        # note that: hops can be either an integer or a list
        # when it's a list, it contains the numbers of hops that you want to test
        # when it's a numebr, it will test all the hops in range(hops) 
        method = method.lower()
        if method not in ['mean', 'max', 'concat', 'simple']:
            raise ValueError("Method not Suppoted! Choose 'mean', 'max' or 'concat' !")    
        
        super(NodeClusteringNAFS, self).__init__()
        self.__dataset = dataset
        self.__labels = self.__dataset.y

        self.__method = method
        self.__r_list = r_list
        self.__hops = range(hops) if type(hops) == int else hops
        self.__seed = seed

        self.__n_clusters = self.__dataset.num_classes
        self.__n_init = n_init

        self.__best_hop_acc, self.__best_hop_nmi, self.__best_hop_adjscore, self.__acc, self.__nmi, self.__adjscore = self._execute()

    @property
    def acc(self):
        return self.__acc
    
    @property
    def nmi(self):
        return self.__nmi
    
    @property
    def adjscore(self):
        return self.__adjscore

    @property
    def best_hop_acc(self):
        return self.__best_hop_acc

    @property
    def best_hop_nmi(self):
        return self.__best_hop_nmi

    @property
    def best_hop_adjscore(self):
        return self.__best_hop_adjscore

    def _execute(self):
        set_seed(self.__seed)

        t_total = time.time()
        best_acc, best_nmi, best_adjscore = 0., 0. ,0.
        best_hop_acc, best_hop_nmi, best_hop_adjscore = 0, 0, 0

        for hop in self.__hops:
            t = time.time()
            acc, nmi, adjscore = self._k_hop_cluster(hop)

            print('hops:{:2d}'.format(hop),
                  'acc: {:.4f}'.format(acc),
                  'nmi: {:.4f}'.format(nmi),
                  'adjscore: {:.4f}'.format(adjscore),
                  'time: {:.4f} seconds'.format(time.time() - t)
                  )

            if acc > best_acc:
                best_acc = acc
                best_hop_acc = hop
            if nmi > best_nmi:
                best_nmi = nmi
                best_hop_nmi = hop
            if adjscore > best_adjscore:
                best_adjscore = adjscore
                best_hop_adjscore = hop

        print("Node Smoothing Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("best_acc: {:.4f}, best_nmi: {:.4f}, best_adjscore: {:.4f}".format(best_acc, best_nmi, best_adjscore))

        return  best_hop_acc, best_hop_nmi, best_hop_adjscore, best_acc, best_nmi, best_adjscore

    def _k_hop_cluster(self, hops):
        input_features = []
        features = torch.tensor(self.__dataset.x, dtype=torch.float)
        r_list = self.__r_list
        for r in r_list:
            t = time.time()
            adj_norm = adj_to_symmetric_norm(self.__dataset.adj, r)
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

        kmeans = KMeans(n_clusters=self.__n_clusters, n_init=self.__n_init, random_state=self.__seed)
        y_pred = kmeans.fit_predict(input_features.numpy())
        cm = clustering_metrics(self.__labels.numpy(), y_pred)
        acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
        return acc, nmi, adjscore
