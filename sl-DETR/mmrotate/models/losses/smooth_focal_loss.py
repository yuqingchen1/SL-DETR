# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from mmdet.models.losses.new_gaussian_loss import new_kld_loss

from mmrotate.registry import MODELS

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
def binary_cross_entropy_loss_with_logits(inputs, pos_weights, neg_weights, avg_factor):
    p = inputs.sigmoid()
    loss = -pos_weights * p.log() - neg_weights * (1-p).log() 
    return loss.sum()/avg_factor



def get_local_rank( quality, indices):
    #quality: one-dimension tensor 
    #indices: matching result
    bs = len(indices)
    device = quality.device
    tgt_size = [len(tgt_ind) for _,tgt_ind in indices]
    ind_start = 0
    rank_list = []
    for i in range(bs):
        if  tgt_size[i] == 0:
            rank_list.append(torch.zeros(0,dtype=torch.long,device=device))
            continue     
        num_tgt = max(indices[i][1]) + 1
        # split quality of one item
        quality_per_img = quality[ind_start:ind_start+tgt_size[i]]
        ind_start += tgt_size[i]
        #suppose candidate bag sizes are equal        
        k = torch.div(tgt_size[i], num_tgt,rounding_mode='floor')
        #sort quality in each candidate bag
        quality_per_img = quality_per_img.reshape(num_tgt, k)
        ind = quality_per_img.sort(dim=-1,descending=True)[1]
        #scatter ranks, eg:[0.3,0.6,0.5] -> [2,0,1]
        rank_per_img = torch.zeros_like(quality_per_img, dtype=torch.long, device = device)
        rank_per_img.scatter_(-1, ind, torch.arange(k,device=device, dtype=torch.long).repeat(num_tgt,1))
        rank_list.append(rank_per_img.flatten())

    return torch.cat(rank_list, 0)

def KL_BCE_loss(src_logits,pos_idx_c, src_boxes, target_boxes, indices, avg_factor, alpha,gamma, w_prime=1, ):
    prob = src_logits.sigmoid()
    #init positive weights and negative weights
    pos_weights = torch.zeros_like(src_logits)
    neg_weights =  prob ** gamma

    # t is the quality metric
    # Control t through KL 
    new_kld_loss = torch.clamp(new_kld_loss, 0, 1)
    t = prob[pos_idx_c] ** self.alpha * new_kld_loss ** (1 - self.alpha)

    t = torch.clamp(t, 0.01).detach()
    rank = get_local_rank(t, indices)
    #define positive weights for SoftBceLoss  
    if type(w_prime) != int:
        rank_weight = w_prime[rank]
    else:
        rank_weight = w_prime
    
    t = t * rank_weight
    pos_weights[pos_idx_c] = t 
    neg_weights[pos_idx_c] = (1 -t)    
    
    loss = -pos_weights * prob.log() - neg_weights * (1-prob).log() 
    return loss.sum()/avg_factor, rank_weight


@MODELS.register_module()
class SmoothFocalLoss(nn.Module):
    """Smooth Focal Loss. Implementation of `Circular Smooth Label (CSL).`__

    __ https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SmoothFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * smooth_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
