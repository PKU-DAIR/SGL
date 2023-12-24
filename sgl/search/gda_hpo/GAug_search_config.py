import torch.nn.functional as F
from openbox import space as osp

from sgl.models.homo.gda import GAugO, GAugM
from sgl.tasks import NodeClassificationGAugO, NodeClassificationGAugM
from sgl.search.gda_hpo.search_config import BaseGDAConfigManager

class GAugOConfigManager(BaseGDAConfigManager):
    def __init__(self, dataset, gnn_type, gae, device, runs=5, activation=F.relu, minibatch=False, epochs=200, max_patience=100, pretrain_batch_size=None, train_batch_size=None):
        super(GAugOConfigManager, self).__init__()
        # basic information
        self.__dataset = dataset
        self.__gnn_type = gnn_type
        self.__gae = gae
        self.__minibatch = minibatch
        self.__activation = activation
        self.__epochs = epochs
        self.__device = device
        self.__max_patience = max_patience
        self.__pretrain_batch_size = pretrain_batch_size
        self.__train_batch_size = train_batch_size
        self.__runs = runs
        self.__config_space = osp.Space()
        # model hyperparameters
        alpha = osp.Real("alpha", 0, 1, default_value=0.4, q=0.01)
        temperature = osp.Real("temperature", 0.1, 2.1, default_value=1.5, q=0.1)
        hidden_dim = osp.Categorical("hidden_dim", [32, 64, 128, 256], default_value=128)
        emb_size = osp.Constant("emb_size", 32)
        n_layers = osp.Constant("n_layers", 2)
        dropout = osp.Constant("dropout", 0.5)
        feat_norm = osp.Constant("feat_norm", "row")
        # task hyperparameters
        lr = osp.Constant("lr", 0.01) if self.__gnn_type != "gat" else osp.Constant("lr", 0.005)
        if pretrain_batch_size is not None:
            ep_lr = osp.Real("ep_lr", 0.001, 0.01, default_value=0.002, q=0.001)
        else:
            ep_lr = osp.Constant("ep_lr", 0.01)
        weight_decay = osp.Constant("weight_decay", 0.0005)
        warmup = osp.Int("warmup", 0, 10, default_value=2, q=1)
        beta = osp.Real("beta", 0, 4, default_value=2, q=0.1)
        pretrain_ep = osp.Int("pretrain_ep", 5, 300, default_value=100, q=5)
        pretrain_nc = osp.Int("pretrain_nc", 5, 300, default_value=200, q=5)
        self.__config_space.add_variables([alpha, temperature, hidden_dim, emb_size, n_layers, dropout, feat_norm, \
                                           lr, weight_decay, warmup, beta, pretrain_ep, pretrain_nc, ep_lr])

    def _configSpace(self):
        return self.__config_space
    
    def _configTarget(self, params):
        model_kwargs = dict()
        model_kwargs["in_dim"] = self.__dataset.num_features
        model_kwargs["hidden_dim"] = params["hidden_dim"]
        model_kwargs["emb_size"] = params["emb_size"]
        model_kwargs["n_classes"] = self.__dataset.num_classes
        model_kwargs["n_layers"] = params["n_layers"]
        model_kwargs["dropout"] = params["dropout"]
        model_kwargs["gnn_type"] = self.__gnn_type
        model_kwargs["activation"] = self.__activation
        model_kwargs["temperature"] = params["temperature"]
        model_kwargs["gae"] = self.__gae
        model_kwargs["alpha"] = params["alpha"]
        model_kwargs["feat_norm"] = params["feat_norm"]
        model_kwargs["minibatch"] = self.__minibatch
        model = GAugO(**model_kwargs)
        task_kwargs = dict()
        task_kwargs["lr"] = params["lr"]
        task_kwargs["weight_decay"] = params["weight_decay"]
        task_kwargs["epochs"] = self.__epochs
        task_kwargs["device"] = self.__device
        task_kwargs["beta"] = params["beta"]
        task_kwargs["warmup"] = params["warmup"]
        task_kwargs["max_patience"] = self.__max_patience
        task_kwargs["pretrain_ep"] = params["pretrain_ep"]
        task_kwargs["pretrain_nc"] = params["pretrain_nc"]
        task_kwargs["pretrain_batch_size"] = self.__pretrain_batch_size
        task_kwargs["train_batch_size"] = self.__train_batch_size
        task_kwargs["ep_lr"] = params["ep_lr"]
        task = NodeClassificationGAugO(self.__dataset, model, runs=self.__runs, verbose=False, **task_kwargs)
        acc_res = task._execute()
        
        return dict(objectives=[-acc_res])

class GAugMConfigManager(BaseGDAConfigManager):
    def __init__(self, dataset, gnn_type, gae, device, num_logits, runs=5, activation=F.relu, epochs=200, max_patience=100):
        super(GAugMConfigManager, self).__init__()
        # basic information
        self.__dataset = dataset
        self.__gnn_type = gnn_type
        self.__gae = gae
        self.__activation = activation
        self.__device = device
        self.__epochs = epochs
        self.__max_patience = max_patience
        self.__runs = runs
        self.__config_space = osp.Space()
        # model hyperparameters 
        choose_idx = osp.Int("choose_idx", 1, num_logits, default_value=1, q=1)
        rm_pct = osp.Int("rm_pct", 0, 80, default_value=20, q=1)
        add_pct = osp.Int("add_pct", 0, 80, default_value=20, q=1)
        hidden_dim = osp.Categorical("hidden_dim", [32, 64, 128, 256], default_value=128)
        n_layers = osp.Constant("n_layers", 2)
        dropout = osp.Constant("dropout", 0.5)
        feat_norm = osp.Constant("feat_norm", "row")
        # task hyperparameters
        lr = osp.Constant("lr", 0.01)
        weight_decay = osp.Constant("weight_decay", 0.0005)
        self.__config_space.add_variables([choose_idx, rm_pct, add_pct, hidden_dim, n_layers, dropout, feat_norm, \
                                           lr, weight_decay])
    
    def _configSpace(self):
        return self.__config_space
    
    def _configTarget(self, params):
        model_kwargs = dict()
        model_kwargs["in_dim"] = self.__dataset.num_features 
        model_kwargs["hidden_dim"] = params["hidden_dim"]
        model_kwargs["n_classes"] = self.__dataset.num_classes
        model_kwargs["n_layers"] = params["n_layers"]
        model_kwargs["dropout"] = params["dropout"]
        model_kwargs["gnn_type"] = self.__gnn_type
        model_kwargs["activation"] = self.__activation
        model_kwargs["gae"] = self.__gae
        model_kwargs["feat_norm"] = params["feat_norm"]
        model_kwargs["choose_idx"] = params["choose_idx"]
        model_kwargs["rm_pct"] = params["rm_pct"]
        model_kwargs["add_pct"] = params["add_pct"]
        model = GAugM(**model_kwargs)
        task_kwargs = dict()
        task_kwargs["lr"] = params["lr"]
        task_kwargs["weight_decay"] = params["weight_decay"]
        task_kwargs["epochs"] = self.__epochs
        task_kwargs["device"] = self.__device
        task_kwargs["max_patience"] = self.__max_patience
        task = NodeClassificationGAugM(self.__dataset, model, runs=self.__runs, verbose=False, **task_kwargs)
        acc_res = task._execute()

        return dict(objectives=[-acc_res])