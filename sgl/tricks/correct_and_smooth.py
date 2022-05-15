import torch
import torch.nn.functional as F
from sgl.tricks.utils import label_propagation


class CorrectAndSmooth:
    def __init__(self, num_correct_layers, correct_alpha, num_smooth_layers, 
                smooth_alpha, autoscale=True, scale=1.0) -> None:
        super().__init__()
        self.__num_correct_layers = num_correct_layers
        self.__correct_alpha = correct_alpha
        self.__num_smooth_layers = num_smooth_layers
        self.__smooth_alpha = smooth_alpha
        self.__autoscale = autoscale
        self.__scale = scale

    # different from pyg implemetation, y_true here represents all the labels for convenience
    @torch.no_grad()
    def correct(self, y_soft, y_true, mask, adj):
        y_soft = y_soft.cpu()
        y_true = y_true.cpu()
        mask = torch.tensor(mask)
        if y_true.dtype == torch.long:
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)

        error = torch.zeros_like(y_soft)
        error[mask] = y_true[mask] - y_soft[mask]
        num_true = mask.shape[0] if mask.dtype == torch.long else int(mask.sum())

        if self.__autoscale:
            smoothed_error = label_propagation(error, adj, self.__num_correct_layers, self.__correct_alpha, post_process=lambda x:x.clamp_(-1., 1.))
            sigma = error[mask].abs().sum() / num_true
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            return y_soft + smoothed_error * scale
        
        else:
            def fix_input(x):
                x[mask] = error[mask]
                return x

            smoothed_error = label_propagation(error, adj, self.__num_correct_layers, self.__correct_alpha, post_process=fix_input)
            return y_soft + smoothed_error * self.__scale 

    @torch.no_grad()
    def smooth(self, y_soft, y_true, mask, adj):
        y_soft = y_soft.cpu()
        y_true = y_true.cpu()
        if y_true.dtype == torch.long:
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)
        
        y_soft[mask] = y_true[mask]

        smoothed_label = label_propagation(y_soft, adj, self.__num_smooth_layers, self.__smooth_alpha)
        return smoothed_label
