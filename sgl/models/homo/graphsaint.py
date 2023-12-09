from sgl.models.simple_models import GCN
from sgl.models.base_model import BaseSAMPLEModel
from sgl.operators.graph_op import RwGraphOp

from torch.nn.functional import nll_loss

class GraphSAINT(BaseSAMPLEModel):
    def __init__(self, dataset, training_sampler, eval_sampler, hidden_dim, dropout=0.5, num_layers=2, device="cpu"):
        super(GraphSAINT, self).__init__()
        self._pre_graph_op = RwGraphOp()
        self._training_sampling_op = training_sampler
        self._eval_sampling_op = eval_sampler
        self.device = device
        self._base_model = GCN(
            n_feat=dataset.num_features, n_hid=hidden_dim, n_class=dataset.num_classes, n_layers=num_layers, dropout=dropout
        ).to(device)

    def pre_sample(self, mode="train"):
        self._training_sampling_op._calc_norm()
        self._training_sampling_op.loss_norm.to(device=self.device)
        return

    def mini_batch_prepare_forward(self, batch, device, **kwargs):
        batch_in, batch_out, block = batch
        local_inds, global_inds = batch_out

        in_x = self._processed_feature[batch_in].to(device)
        y_truth = self._vanilla_y[global_inds].to(device)
        block.to_device(device)
        y_pred = self._base_model(in_x, block)[local_inds]
        return y_pred, y_truth
    
    def collate_fn(self, batch_ids, mode):
        if mode == "train":
            return self._training_sampling_op.collate_fn(batch_ids, mode)
        else:
            return self._eval_sampling_op.collate_fn(batch_ids, mode)

    def loss(self, pred, labels):
        loss = nll_loss(pred, labels, reduction="none")
        loss = (loss / self.cur_loss_norm).sum()
        return loss

    @property
    def cur_loss_norm(self):
        return self._training_sampling_op.loss_norm[self._training_sampling_op.cur_index]