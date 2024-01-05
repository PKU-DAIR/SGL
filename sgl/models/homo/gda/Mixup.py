import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv

from sgl.data.base_data import Block
from sgl.models.base_model import BaseSAMPLEModel

class Mixup(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout, alpha, beta, gnn_type="sage", feat_norm="row", activation=F.relu, **kwargs):
        super(Mixup, self).__init__()
        self.__alpha = alpha 
        self.__beta = beta 
        self.__feat_norm = feat_norm
        self.nc_net = TwoBranchGNN(in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, activation, **kwargs)

    def preprocess(self, adj, features, device):
        if self.__feat_norm == "row":
            features = F.normalize(features, p=1, dim=1)
        self.__num_nodes = features.size(0)
        self.__features = features.to(device)
        if isinstance(adj, sp.coo_matrix) is False:
            adj = sp.coo_matrix(adj)
        adj.setdiag(0)
        self.__row = torch.from_numpy(adj.row).to(torch.long)
        self.__col = torch.from_numpy(adj.col).to(torch.long)
        self.__adj = torch.vstack([self.__row, self.__col]).to(device)
    
    @property
    def processed_feature(self):
        return self.__features
    
    @property 
    def processed_block(self):
        return self.__adj
    
    @staticmethod 
    def loss_fn(mix_ratio, output, y_raw, y_b, train_idx):
        loss = F.nll_loss(output[train_idx], y_raw[train_idx]) * mix_ratio + \
                F.nll_loss(output[train_idx], y_b[train_idx]) * (1 - mix_ratio)
        return loss

    def reset_parameters(self):
        self.nc_net.reset_parameters()

    def train_func(self, train_idx, y_raw, device, optimizer, loss_fn, metric):
        self.nc_net.train()
        mix_ratio = np.random.beta(self.__alpha, self.__beta) 
        id_old_value_new, adj_b, y_b = self._mixup(train_idx, y_raw, device)  
        output = self.nc_net(self.__features, self.__adj, adj_b, mix_ratio, id_old_value_new)

        loss = loss_fn(mix_ratio, output, y_raw, y_b, train_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.nc_net.eval()
        output = self.forward(self.__features, self.__adj)
        acc = metric(output[train_idx], y_raw[train_idx])

        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate_func(self, val_idx, test_idx, labels, device, metric):
        self.nc_net.eval()

        pred_y = self.forward(self.__features, self.__adj)
        
        acc_val = metric(pred_y[val_idx], labels[val_idx])
        acc_test = metric(pred_y[test_idx], labels[test_idx])
        
        return acc_val, acc_test
    
    def _mixup(self, train_idx, y_raw, device):
        id_old_value_new = torch.arange(self.__num_nodes, dtype=torch.long)
        train_idx_shuffle = np.asarray(train_idx)
        np.random.shuffle(train_idx_shuffle)
        # map raw node id to its pair node id
        id_old_value_new[train_idx] = torch.from_numpy(train_idx_shuffle).to(torch.long)
        id_new_value_old = torch.zeros_like(id_old_value_new)
        # map the pair node id to the raw node id
        id_new_value_old[id_old_value_new] = torch.arange(self.__num_nodes, dtype=torch.long)
        row_b = id_old_value_new[self.__row]
        col_b = id_old_value_new[self.__col]
        adj_b = torch.vstack([row_b, col_b]).to(device)
        y_b = y_raw[id_old_value_new]

        return id_old_value_new, adj_b, y_b
    
    def model_forward(self, idx, device):
        output = self.forward(self.__features, self.__adj)
        
        return output[idx]
    
    def forward(self, x, adj):
        output = self.nc_net(x, adj, adj, 1, np.arange(self.__num_nodes))

        return output
    
    def postprocess(self, adj, output):
        return output
    

class SampleMixup(BaseSAMPLEModel):
    def __init__(self, training_sampler, eval_sampler, in_dim, hidden_dim, n_classes, n_layers, dropout, alpha, beta, gnn_type="sage", feat_norm="row", activation=F.relu, **kwargs):
        super(SampleMixup, self).__init__(sparse_type="pyg")
        self.__alpha = alpha 
        self.__beta = beta 
        self.__feat_norm = feat_norm
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = MinibatchTwoBranchGNN(in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, activation, **kwargs)

    def preprocess(self, adj, x, y, device, **kwargs):
        if self.__feat_norm == "row":
            x = F.normalize(x, p=1, dim=1)
        self.__num_nodes = x.size(0)
        self.__features = x.to(device)
        if isinstance(adj, sp.coo_matrix) is False:
            adj = sp.coo_matrix(adj)
        adj.setdiag(0)
        self.__adj = Block(adj, sparse_type="pyg")

        self.__vanilla_y = y

        inductive = kwargs.get("inductive", False)
        if inductive is True:
            train_idx = kwargs.get("train_idx", None)
            if train_idx is None:
                raise ValueError(f"For inductive learning, "
                                 "please pass train idx "
                                 "as the parameters of preprocess function.")
            self.__train_features = x[train_idx]
            self.__vanilla_train_y = y[train_idx]

    @property
    def processed_feature(self):
        return self.__features
    
    @property 
    def processed_block(self):
        return self.__adj
    
    @staticmethod 
    def loss_fn(mix_ratio, output, y_raw, y_b):
        loss = F.nll_loss(output, y_raw) * mix_ratio + \
                F.nll_loss(output, y_b) * (1 - mix_ratio)
        return loss

    def mini_batch_prepare_forward(self, batch, device, loss_fn, optimizer, inductive=False, transfer_y_to_device=True, mix_ratio=1):
        batch_in, batch_out, block = batch 

        if inductive is False:
            in_x = self.__features[batch_in].to(device)
            y_raw = self.__vanilla_y[batch_out]
        else:
            in_x = self.__train_features[batch_in].to(device)
            y_raw = self.__vanilla_train_y[batch_out]

        if transfer_y_to_device is True:
            y_raw = y_raw.to(device)
        
        id_old_value_new, block_b, y_b = self._mixup(batch_out.shape[0], batch_in.shape[0], block, y_raw)  
        block.to_device(device)
        block_b.to_device(device)
        output = self._base_model(in_x, block, block_b, mix_ratio, id_old_value_new)

        loss = loss_fn(mix_ratio, output, y_raw, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        return loss.item(), output, y_raw 
    
    def train_func(self, train_loader, inductive, device, optimizer, loss_fn):
        correct_num = 0
        loss_train_sum = 0.
        train_num = 0

        self._base_model.train()
        mix_ratio = np.random.beta(self.__alpha, self.__beta) 
        
        for batch in train_loader:
            loss_train, y_out, y_truth = self.mini_batch_prepare_forward(batch, device, loss_fn, optimizer, inductive=inductive, mix_ratio=mix_ratio)
            pred = y_out.max(1)[1].type_as(y_truth)
            correct_num += pred.eq(y_truth).double().sum()
            loss_train_sum += loss_train
            train_num += len(y_truth)
        
        loss_train = loss_train_sum / len(train_loader)
        acc_train = correct_num / train_num

        return loss_train, acc_train.item()
    
    @torch.no_grad()
    def evaluate_func(self, val_loader, test_loader, device):
        self._base_model.eval()

        correct_num_val, correct_num_test = 0, 0
        val_num = 0
        for batch in val_loader:
            val_output, out_y = self.model_forward(batch, device)
            pred = val_output.max(1)[1].type_as(out_y)
            correct_num_val += pred.eq(out_y).double().sum()
            val_num += len(out_y)
            
        acc_val = correct_num_val / val_num

        test_num = 0
        for batch in test_loader:
            test_output, out_y = self.model_forward(batch, device)
            pred = test_output.max(1)[1].type_as(out_y)
            correct_num_test += pred.eq(out_y).double().sum()
            test_num += len(out_y)
        
        acc_test = correct_num_test / test_num

        return acc_val.item(), acc_test.item()
        
    def _mixup(self, num_train_nodes, batch_size, block, y_raw):
        id_old_value_new = torch.arange(batch_size, dtype=torch.long)
        train_idx_shuffle = np.arange(num_train_nodes)
        np.random.shuffle(train_idx_shuffle)
        # map raw node id to its pair node id
        id_old_value_new[:num_train_nodes] = torch.from_numpy(train_idx_shuffle).to(torch.long)
        id_new_value_old = torch.zeros_like(id_old_value_new)
        # map the pair node id to the raw node id
        id_new_value_old[id_old_value_new] = torch.arange(batch_size, dtype=torch.long)
        adjs_b = []
        for i in range(len(block)):
            adj = block[i]
            if isinstance(adj, sp.coo_matrix) is False:
                adj = sp.coo_matrix(adj)
            row, col = adj.row, adj.col
            row_b = id_old_value_new[row]
            col_b = id_old_value_new[col]
            adj_b = SparseTensor(row=row_b, col=col_b, value=torch.ones_like(row_b))
            adjs_b.append(adj_b)
       
        block_b = Block(adjs_b, sparse_type="pyg")
        
        y_b = y_raw[train_idx_shuffle]

        return id_old_value_new, block_b, y_b

    def postprocess(self, adj, output):
        return output
    
    def model_forward(self, batch_in, block, device):
        x = self.__features[batch_in].to(device)
        block.to_device(device)
        output = self.forward(x, block)
        
        return output
    
    def forward(self, x, block):
        output = self._base_model(x, block, block, 1, np.arange(self.__num_nodes))

        return output 

class TwoBranchGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, activation=F.relu, **kwargs):
        super(TwoBranchGNN, self).__init__()
        self.gcs = nn.ModuleList()
        if gnn_type != "sage":
            raise NotImplementedError
        self.gcs.append(SAGEConv(in_dim, hidden_dim))
        self.batch_norm = kwargs.get("batch_norm", False)
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers-1):
            self.gcs.append(SAGEConv(hidden_dim, hidden_dim))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin = nn.Linear(hidden_dim, n_classes)
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for gc in self.gcs:
            gc.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x0, adj, adj_b, mix_ratio, id_old_value_new):
        aggr_xs = [x0]
        for i in range(self.n_layers-1):
            x = self.gcs[i](aggr_xs[-1], adj)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            aggr_xs.append(x)

        aggr_xs_b = []
        for x in aggr_xs:
            aggr_xs_b.append(x[id_old_value_new])

        x_mix = aggr_xs[0] * mix_ratio + aggr_xs_b[0] * (1 - mix_ratio)
        for i in range(self.n_layers):
            x_new = self.gcs[i]((aggr_xs[i], x_mix), adj)
            if self.batch_norm:
                x_new = self.bns[i](x_new)
            x_new = self.activation(x_new)
            
            x_new_b = self.gcs[i]((aggr_xs_b[i], x_mix), adj_b)
            if self.batch_norm:
                x_new_b = self.bns[i](x_new_b)
            x_new_b = self.activation(x_new_b)
            
            x_mix = x_new * mix_ratio + x_new_b * (1 - mix_ratio)
            x_mix = F.dropout(x_mix, self.dropout, training=self.training)

        x = self.lin(x_mix)
        return F.log_softmax(x, dim=-1)

class MinibatchTwoBranchGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, activation=F.relu, **kwargs):
        super(MinibatchTwoBranchGNN, self).__init__()
        self.gcs = nn.ModuleList()
        if gnn_type != "sage":
            raise NotImplementedError
        self.gcs.append(SAGEConv(in_dim, hidden_dim))
        self.batch_norm = kwargs.get("batch_norm", False)
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers-1):
            self.gcs.append(SAGEConv(hidden_dim, hidden_dim))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin = nn.Linear(hidden_dim, n_classes)
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for gc in self.gcs:
            gc.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x0, block, block_b, mix_ratio, id_old_value_new):
        aggr_xs = [x0]
        for i in range(self.n_layers):
            root_size = block.root_size(i)
            root_x = aggr_xs[-1][:root_size]
            x = self.gcs[i]((aggr_xs[-1], root_x), block[i])
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            aggr_xs.append(x)

        aggr_xs_b = []
        for x in aggr_xs:
            num_nodes = x.size(0)
            aggr_xs_b.append(x[id_old_value_new[:num_nodes]])

        x_mix = aggr_xs[0] * mix_ratio + aggr_xs_b[0] * (1 - mix_ratio)
        for i in range(self.n_layers):
            root_size = block.root_size(i)
            root_x = x_mix[:root_size]
            x_new = self.gcs[i]((aggr_xs[i], root_x), block[i])
            if self.batch_norm:
                x_new = self.bns[i](x_new)
            x_new = self.activation(x_new)
            
            root_size = block_b.root_size(i)
            root_x_b = x_mix[:root_size]
            x_new_b = self.gcs[i]((aggr_xs_b[i], root_x_b), block_b[i])
            if self.batch_norm:
                x_new_b = self.bns[i](x_new_b)
            x_new_b = self.activation(x_new_b)
            x_mix = x_new * mix_ratio + x_new_b * (1 - mix_ratio)
            x_mix = F.dropout(x_mix, self.dropout, training=self.training)

        x = self.lin(x_mix)
        return F.log_softmax(x, dim=-1)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
         
        for i in range(self.n_layers):
            xs = []
            for batch in subgraph_loader:
                batch_in, batch_out, block = batch
                block.to_device(device)
                x = x_all[batch_in].to(device)
                root_size = len(batch_out)
                root_x = x[:root_size]
                x = self.gcs[i]((x, root_x), block[0]) # one-layer sampling            
                if self.batch_norm:
                    x = self.bns[i](x)   
                x = self.activation(x)
                if i == self.n_layers-1:
                    x = self.lin(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        
        return x_all

    