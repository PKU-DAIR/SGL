import torch
import torch.nn as nn
import torch.nn.functional as F

from sgl.models.base_model import BaseSAMPLEModel
from sgl.utils import sparse_mx_to_pyg_sparse_tensor
from sgl.models.pyg_simple_models import GCN, SAGE, GAT

GNN_BACKBONE = {"gcn": GCN, "sage": SAGE, "gat": GAT}

class FLAG(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, step_size, augM, activation=F.relu, **kwargs):
        super(FLAG, self).__init__()
        self.__step_size = float(step_size) 
        self.__augM = augM
        self.__amp = kwargs.pop("amp", 1)
        if isinstance(activation, str):
            activation = getattr(F, activation)
        if gnn_type == 'gat':
            if kwargs.get("n_heads"):
                n_heads = list(map(lambda x: int(x), kwargs["n_heads"].split(",")))
            else:
                n_heads = [8] * (n_layers - 1) + [1]
            kwargs.update({"n_heads": n_heads})
        self._base_model = GNN_BACKBONE.get(gnn_type)(in_dim, hidden_dim, n_classes, n_layers=n_layers, dropout=dropout, activation=activation, **kwargs)

    @property
    def processed_feature(self):
        return self.__features
    
    @property
    def processed_adj(self):
        return self.__processed_adj
    
    def preprocess(self, adj, features, device):
        self.__features = features.to(device)
        self.__processed_adj = sparse_mx_to_pyg_sparse_tensor(adj).to(device)

    def flag(self, ground_truth_y, optimizer, device, train_idx, loss_fn):
        x = self.__features
        adj = self.__processed_adj
        
        self._base_model.train()
        optimizer.zero_grad()

        perturb = torch.FloatTensor(x.shape).uniform_(-self.__step_size, self.__step_size).to(device)
        unlabel_idx = list(set(range(perturb.shape[0])) - set(train_idx))
        perturb.data[unlabel_idx] *= self.__amp
        
        perturb.requires_grad_()
        pred_y = self._base_model(x+perturb, adj)[train_idx]
        loss = loss_fn(pred_y, ground_truth_y)
        loss /= self.__augM

        for _ in range(self.__augM-1):
            loss.backward()
            perturb_data = perturb[train_idx].detach() + self.__step_size * torch.sign(perturb.grad[train_idx].detach())
            perturb.data[train_idx] = perturb_data.data
            perturb_data = perturb[unlabel_idx].detach() + self.__amp * self.__step_size * torch.sign(perturb.grad[unlabel_idx].detach())
            perturb.data[unlabel_idx] = perturb_data.data
            perturb.grad[:] = 0 

            pred_y = self._base_model(x+perturb, adj)[train_idx]
            loss = loss_fn(pred_y, ground_truth_y)
            loss /= self.__augM
        
        loss.backward()
        optimizer.step()

        return loss.item()

    @staticmethod
    def model_train(model, train_idx, labels, device, optimizer, loss_fn, metric):
        loss_train = model.flag(labels[train_idx], optimizer, device, train_idx, loss_fn)

        model.eval()
        pred_y = model(model.processed_feature, model.processed_adj)
        acc_train = metric(pred_y[train_idx], labels[train_idx])

        return loss_train, acc_train
    
    @staticmethod
    @torch.no_grad()
    def model_evaluate(model, val_idx, test_idx, labels, device, metric):
        model.eval()
        pred_y = model(model.processed_feature, model.processed_adj)

        acc_val = metric(pred_y[val_idx], labels[val_idx])
        acc_test = metric(pred_y[test_idx], labels[test_idx])
        return acc_val, acc_test
    
    def model_forward(self, idx, device):
        pred_y = self.forward(self.__features, self.__processed_adj)
        return pred_y[idx]

    def forward(self, x, adj):
        return self._base_model(x, adj)
    
    def postprocess(self, adj, outputs):
        return outputs
    

class SampleFLAG(BaseSAMPLEModel):
    def __init__(self, training_sampler, eval_sampler, in_dim, hidden_dim, n_classes, n_layers, dropout, gnn_type, step_size, augM, activation=F.relu, **kwargs):
        super(SampleFLAG, self).__init__()
        self.__step_size = float(step_size) 
        self.__augM = augM
        self.__amp = kwargs.pop("amp", 1)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        if isinstance(activation, str):
            activation = getattr(F, activation)
        if gnn_type == 'gat':
            if kwargs.get("n_heads"):
                n_heads = list(map(lambda x: int(x), kwargs["n_heads"].split(",")))
            else:
                n_heads = [8] * (n_layers - 1) + [1]
            kwargs.update({"n_heads": n_heads})
        self._base_model = GNN_BACKBONE.get(gnn_type)(in_dim, hidden_dim, n_classes, n_layers=n_layers, dropout=dropout, activation=activation, **kwargs)

    def flag(self, clean, ground_truth_y, adjs, batch_out, optimizer, device, loss_fn):
        self._base_model.train()
        optimizer.zero_grad()
        batch_size = len(batch_out)
        perturb_t = torch.FloatTensor(clean[:batch_size].shape).uniform_(-self.__step_size, self.__step_size).to(device)
        perturb_un = torch.FloatTensor(clean[batch_size:].shape).uniform_(-self.__amp * self.__step_size, self.__amp * self.__step_size).to(device)
        perturb_t.requires_grad_()
        perturb_un.requires_grad_()
        
        perturb = torch.cat([perturb_t, perturb_un], dim=0)
        pred_y = self._base_model(clean+perturb, adjs)
        loss = loss_fn(pred_y, ground_truth_y)
        loss /= self.__augM

        for _ in range(self.__augM-1):
            loss.backward()
            
            perturb_data_t = perturb_t.detach() + self.__step_size * torch.sign(perturb_t.grad.detach())
            perturb_t.data = perturb_data_t.data
            perturb_t.grad[:] = 0

            perturb_data_un = perturb_un.detach() + self.__amp * self.__step_size * torch.sign(perturb_un.grad.detach())
            perturb_un.data = perturb_data_un.data
            perturb_un.grad[:] = 0

            perturb = torch.cat((perturb_t, perturb_un), dim=0)

            pred_y = self._base_model(clean+perturb, adjs)
            loss = loss_fn(pred_y, ground_truth_y)
            loss /= self.__augM
        
        loss.backward()
        optimizer.step()

        return loss.item(), pred_y

    def mini_batch_prepare_forward(self, batch, device, loss_fn, optimizer, inductive=False, transfer_y_to_device=True):
        batch_in, batch_out, block = batch
        
        if inductive is False:
            in_x = self._processed_feature[batch_in].to(device)
            y_truth = self._vanilla_y[batch_out]
        else:
            in_x = self._processed_train_feature[batch_in].to(device)
            y_truth = self._vanilla_train_y[batch_out]
        
        if transfer_y_to_device is True:
            y_truth = y_truth.to(device)
        
        block.to_device(device)
        loss, pred_y = self.flag(in_x, y_truth, block, batch_out, optimizer, device, loss_fn)

        return loss, pred_y, y_truth
    
    @staticmethod
    def model_train(model, train_loader, inductive, device, optimizer, loss_fn):
        correct_num = 0
        loss_train_sum = 0.
        train_num = 0

        for batch in train_loader:
            loss_train, y_out, y_truth = model.mini_batch_prepare_forward(batch, device, loss_fn, optimizer, inductive=inductive)
            pred = y_out.max(1)[1].type_as(y_truth)
            correct_num += pred.eq(y_truth).double().sum()
            loss_train_sum += loss_train
            train_num += len(y_truth)

        loss_train = loss_train_sum / len(train_loader)
        acc_train = correct_num / train_num

        return loss_train, acc_train.item()
    
    @staticmethod
    @torch.no_grad()
    def model_evaluate(model, val_loader, test_loader, device):
        model.eval()

        correct_num_val, correct_num_test = 0, 0
        val_num = 0
        for batch in val_loader:
            val_output, out_y = model.model_forward(batch, device)
            pred = val_output.max(1)[1].type_as(out_y)
            correct_num_val += pred.eq(out_y).double().sum()
            val_num += len(out_y)
            
        acc_val = correct_num_val / val_num

        test_num = 0
        for batch in test_loader:
            test_output, out_y = model.model_forward(batch, device)
            pred = test_output.max(1)[1].type_as(out_y)
            correct_num_test += pred.eq(out_y).double().sum()
            test_num += len(out_y)
        
        acc_test = correct_num_test / test_num

        return acc_val.item(), acc_test.item()
    
    def model_forward(self, batch, device):
        batch_in, batch_out, block = batch
        in_x = self._processed_feature[batch_in].to(device)
        y_truth = self._vanilla_y[batch_out].to(device)
        block.to_device(device)

        y_pred = self.forward(in_x, block)
        return y_pred, y_truth

    def forward(self, x, adj):
        return self._base_model(x, adj)
    
    def postprocess(self, adj, outputs):
        return outputs