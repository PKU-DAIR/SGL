import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_layers=2, dropout=0.5, activation=F.relu, batch_norm=False, add_self_loops=True, normalize=True, cached=False):
        super(GCN, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(GCNConv(n_feat, n_hid, cached=cached, add_self_loops=add_self_loops, normalize=normalize))
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(n_hid))
        self.n_layers = n_layers
        for _ in range(n_layers-2):
            self.gcs.append(GCNConv(n_hid, n_hid, cached=cached, add_self_loops=add_self_loops, normalize=normalize))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(n_hid))
        self.gcs.append(GCNConv(n_hid, n_class, cached=cached, add_self_loops=add_self_loops, normalize=normalize))
        self.dropout = dropout
        self.activation = activation
    
    def reset_parameters(self):
        for conv in self.gcs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, block):
        repr = x
        if isinstance(block, (SparseTensor, torch.Tensor)):
            block = [block]
        if len(block) == self.n_layers:
            for i in range(self.n_layers-1):
                repr = self.gcs[i](repr, block[i])
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[-1])
        elif len(block) == 1:
            for i in range(self.n_layers-1):
                repr = self.gcs[i](repr, block[0])
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[0])
        else:
            raise ValueError('The sampling layer must be equal to GNN layer.')
        
        return F.log_softmax(repr, dim=1)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.n_layers):
            xs = []
            for batch in subgraph_loader:
                batch_in, _, block = batch
                block.to_device(device)
                x = x_all[batch_in].to(device)
                x = self.gcs[i](x, block[0]) # one-layer sampling            
                if i != self.n_layers - 1:
                    if self.batch_norm:
                        x = self.bns[i](x)   
                    x = self.activation(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
    
class SAGE(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_layers=2, dropout=0.5, activation=F.relu, batch_norm=False, normalize=False):
        super(SAGE, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(SAGEConv(n_feat, n_hid))
        self.batch_norm = batch_norm
        self.normalize = normalize
        if normalize:
            self.norm = lambda x: F.normalize(x, p=1, dim=1)
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(n_hid))
        self.n_layers = n_layers
        for _ in range(n_layers-2):
            self.gcs.append(SAGEConv(n_hid, n_hid))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(n_hid))
        self.gcs.append(SAGEConv(n_hid, n_class))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.gcs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, block):
        repr = x
        if isinstance(block, (SparseTensor, torch.Tensor)):
            block = [block]
        if len(block) == self.n_layers:
            for i in range(self.n_layers-1):
                root_size = block.root_size(i)
                root_repr = repr[:root_size]
                repr = self.gcs[i]((repr, root_repr), block[i])
                if self.normalize:
                    repr = self.norm(repr)
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            root_size = block.root_size(-1)
            root_repr = repr[:root_size]
            repr = self.gcs[-1]((repr, root_repr), block[-1])
        elif len(block) == 1:
            for i in range(self.n_layers-1):
                repr = self.gcs[i](repr, block[0])
                if self.normalize:
                    repr = self.norm(repr)
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[0])
        else:
            raise ValueError('The sampling layer must be equal to GNN layer.')
        
        return F.log_softmax(repr, dim=1)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.n_layers):
            xs = []
            for batch in subgraph_loader:
                batch_in, batch_out, block = batch
                block.to_device(device)
                x = x_all[batch_in].to(device)
                root_size = len(batch_out)
                root_x = x[:root_size]
                x = self.gcs[i]((x, root_x), block[0]) 
                # one-layer sampling
                if i != self.n_layers - 1:
                    if self.batch_norm:
                        x = self.bns[i](x)   
                    x = self.activation(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
    
class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads, n_layers=2, dropout=0.6, activation=F.elu, attn_dropout=0.6, batch_norm=False):
        super(GAT, self).__init__()
        self.gcs = nn.ModuleList()       
        self.gcs.append(GATConv(n_feat, n_hid // n_heads[0], n_heads[0], dropout=attn_dropout))
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(n_hid))
        for i in range(n_layers-2):
            self.gcs.append(GATConv(n_hid, n_hid // n_heads[i + 1], n_heads[i + 1], dropout=attn_dropout))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(n_hid))
        self.gcs.append(GATConv(n_hid, n_class, n_heads[-1], concat=False, dropout=attn_dropout))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.gcs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, block):
        repr = x
        if isinstance(block, (SparseTensor, torch.Tensor)):
            block = [block]
        if len(block) == self.n_layers:
            for i in range(self.n_layers-1):
                root_size = block.root_size(i)
                root_repr = repr[:root_size]
                repr = self.gcs[i]((repr, root_repr), block[i])
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            root_size = block.root_size(-1)
            root_repr = repr[:root_size]
            repr = self.gcs[-1]((repr, root_repr), block[-1])
        elif len(block) == 1:
            for i in range(self.n_layers-1):
                repr = self.gcs[i](repr, block[0])
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[0])
        else:
            raise ValueError('The sampling layer must be equal to GNN layer.')
        
        return F.log_softmax(repr, dim=-1)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.n_layers):
            xs = []
            for batch in subgraph_loader:
                batch_in, batch_out, block = batch
                block.to_device(device)
                x = x_all[batch_in].to(device)
                root_size = len(batch_out)
                root_x = x[:root_size]
                x = self.gcs[i]((x, root_x), block[0]) 
                # one-layer sampling
                if i != self.n_layers - 1:
                    if self.batch_norm:
                        x = self.bns[i](x)   
                    x = self.activation(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all