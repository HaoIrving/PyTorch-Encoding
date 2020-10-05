import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from .ohem_pixel_sampler import OHEMPixelSampler

__all__ = ['LabelSmoothing', 'NLLMultiLabelSmooth', 'SegmentationLosses']

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
    
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
    
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, OHEM=False,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        self.OHEM = OHEM
        if OHEM:
            self.sampler = OHEMPixelSampler(thresh=0.7, min_kept=100000)
            super(SegmentationLosses, self).__init__(weight, None, ignore_index, reduction='none')
        else:
            super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 


    def forward(self, *inputs):
        if self.OHEM:
            if not self.se_loss: # psp deeplab
                pred1, pred2, target = tuple(inputs)
                loss1 = self.OHEMlosses(pred1, target)
                loss2 = self.OHEMlosses(pred2, target)
                return loss1 + self.aux_weight * loss2
            else: # encnet
                pred1, se_pred, pred2, target = tuple(inputs)
                se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
                loss1 = self.OHEMlosses(pred1, target)
                loss2 = self.OHEMlosses(pred2, target)
                loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
                return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

        if not self.OHEM:
            if not self.se_loss and not self.aux:
                return super(SegmentationLosses, self).forward(*inputs)
            elif not self.se_loss:
                pred1, pred2, target = tuple(inputs)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                return loss1 + self.aux_weight * loss2
            elif not self.aux:
                pred, se_pred, target = tuple(inputs)
                se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
                loss1 = super(SegmentationLosses, self).forward(pred, target)
                loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
                return loss1 + self.se_weight * loss2
            else:
                pred1, se_pred, pred2, target = tuple(inputs)
                se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
                return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    def OHEMlosses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = super(SegmentationLosses, self).forward(seg_logit, seg_label)
        seg_label = seg_label.unsqueeze(1)
        seg_weight = self.sampler.sample(seg_logit, seg_label)
        # apply weights and do the reduction, init is 'none'
        seg_weight = seg_weight.float()
        loss = weight_reduce_loss(
            loss, weight=seg_weight, reduction='mean', avg_factor=None)
        
        return loss

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect



def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
