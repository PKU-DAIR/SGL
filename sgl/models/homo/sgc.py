from sgl.models.base_model import BaseSGAPModel
from sgl.models.simple_models import LogisticRegression
from sgl.operators.graph_op import LaplacianGraphOp
from sgl.operators.message_op import LastMessageOp


class SGC(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(SGC, self).__init__(prop_steps, feat_dim, output_dim)
        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5) ###正则化+传播
        self._pre_msg_op = LastMessageOp() #拼接等多hop操作
        self._base_model = LogisticRegression(feat_dim, output_dim)
