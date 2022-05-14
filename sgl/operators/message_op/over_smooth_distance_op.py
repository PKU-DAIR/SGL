import torch
import torch.nn.functional as F

from sgl.operators.base_op import MessageOp

class OverSmoothDistanceWeightedOp(MessageOp):
    def __init__(self):
        super(OverSmoothDistanceWeightedOp, self).__init__()
        self._aggr_type = 'over_smooth_dis_weighted'

    def _combine(self, feat_list): 
        weight_list = []
        features = feat_list[0]
        norm_fea = torch.norm(features, 2, 1).add(1e-10)
        for fea in feat_list:
            norm_cur = torch.norm(fea, 2, 1).add(1e-10)
            tmp = torch.div((features * fea).sum(1), norm_cur)
            tmp = torch.div(tmp, norm_fea)

            weight_list.append(tmp.unsqueeze(-1))

        weight = F.softmax(torch.cat(weight_list, dim=1), dim=1)

        hops = len(feat_list)
        num_nodes = features.shape[0]
        output = []
        for i in range(num_nodes):
            fea = 0.
            for j in range(hops):
                fea += (weight[i][j]*feat_list[j][i]).unsqueeze(0)
            output.append(fea)
        output = torch.cat(output, dim=0)
        return output 
