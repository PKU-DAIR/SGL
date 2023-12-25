from sgl.models.pyg_simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel

class ClusterGCN(BaseSAMPLEModel):
    def __init__(self, training_sampler, eval_sampler, nfeat, hidden_dim, nclass, sparse_type="torch", dropout=0.5, num_layers=2, device="cpu"):
        super(ClusterGCN, self).__init__(evaluate_mode="sampling", sparse_type=sparse_type)
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self._base_model = GCN(n_feat=nfeat, n_hid=hidden_dim, n_class=nclass, n_layers=num_layers, dropout=dropout).to(device)

    def pre_sample(self, mode="train"):
        if mode == "train":
            self._training_sampling_op.multiple_graphs_sampling()
        else:
            self._eval_sampling_op.multiple_graphs_sampling()

    def mini_batch_prepare_forward(self, batch, device, inductive=False):
        batch_in, batch_out, block = batch
        local_inds, global_inds = batch_out
        
        if inductive is False:
            in_x = self._processed_feature[batch_in].to(device)
            y_truth = self._vanilla_y[global_inds].to(device)
        else:
            in_x = self._processed_train_feature[batch_in].to(device)
            y_truth = self._vanilla_train_y[global_inds].to(device)
        
        block.to_device(device)
        y_pred = self._base_model(in_x, block)[local_inds]
        return y_pred, y_truth
    
    def collate_fn(self, batch_inds, mode):
        if self.training:
            return self._training_sampling_op.collate_fn(batch_inds, mode)
        else:
            return self._eval_sampling_op.collate_fn(batch_inds, mode)
