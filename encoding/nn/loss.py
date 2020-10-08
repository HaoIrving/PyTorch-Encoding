import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from .ohem_pixel_sampler import OHEMPixelSampler
import numpy as np

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


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FwIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(FwIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).cuda(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W
        #freq为各类的比例
        freq = np.array([0.04272119, 0.10241207, 0.13548531, 0.28421111, 0.21262745, 0.11806116, 0.10448171])
        freq = torch.from_numpy(freq).float()
        freq = freq.cuda(target.device)

        N = input.shape[0]

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = - inter / (union + 1e-16)
        fwIoU = (loss * freq).sum()
        # Return average loss over classes and batch
        return fwIoU


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, OHEM=False, ohemprob=False, ohemth=0.7,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        self.OHEM = OHEM
        if OHEM and ohemprob:
            self.sampler = OHEMPixelSampler(thresh=ohemth, min_kept=100000) 
            super(SegmentationLosses, self).__init__(weight, None, ignore_index, reduction='none')
        if OHEM and not ohemprob:
            self.sampler = OHEMPixelSampler(thresh=None, min_kept=100000) 
            super(SegmentationLosses, self).__init__(weight, None, ignore_index, reduction='none')
        if not OHEM:
            super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 
        self.ohem_weight = 0.1
        self.FwIoULoss = FwIoULoss(nclass)


    def forward(self, *inputs):
        if self.OHEM:
            if not self.se_loss and not self.aux:
                ohemloss = self.OHEMlosses(*inputs)
                return super(SegmentationLosses, self).forward(*inputs).mean() + self.ohem_weight * ohemloss
            elif not self.se_loss: # psp deeplab
                pred1, pred2, target = tuple(inputs)
                ohemloss1 = self.OHEMlosses(pred1, target)
                ohemloss2 = self.OHEMlosses(pred2, target)
                loss1 = super(SegmentationLosses, self).forward(pred1, target).mean() + self.ohem_weight * ohemloss1
                loss2 = super(SegmentationLosses, self).forward(pred2, target).mean() + self.ohem_weight * ohemloss2
                fwiouloss1 = self.FwIoULoss.forward(pred1, target)
                fwiouloss2 = self.FwIoULoss.forward(pred2, target)
                return loss1 + self.aux_weight * loss2 + 0.3 * (fwiouloss1 + fwiouloss2)
            elif not self.aux:
                pred, se_pred, target = tuple(inputs)
                se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
                ohemloss1 = self.OHEMlosses(pred, target)
                loss1 = super(SegmentationLosses, self).forward(pred, target).mean() + self.ohem_weight * ohemloss1
                loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
                return loss1 + self.se_weight * loss2
            else: # encnet
                pred1, se_pred, pred2, target = tuple(inputs)
                se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
                ohemloss1 = self.OHEMlosses(pred1, target)
                ohemloss2 = self.OHEMlosses(pred2, target)
                loss1 = super(SegmentationLosses, self).forward(pred1, target).mean() + self.ohem_weight * ohemloss1
                loss2 = super(SegmentationLosses, self).forward(pred2, target).mean() + self.ohem_weight * ohemloss2
                loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
                return loss1 + self.aux_weight * loss2 + self.se_weight * loss3
        if not self.OHEM:
            if not self.se_loss and not self.aux:
                return super(SegmentationLosses, self).forward(*inputs)
            elif not self.se_loss:
                pred1, pred2, target = tuple(inputs)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                fwiouloss1 = self.FwIoULoss.forward(pred1, target)
                fwiouloss2 = self.FwIoULoss.forward(pred2, target)
                return loss1 + self.aux_weight * loss2 + 0.3 * (fwiouloss1 + fwiouloss2)
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
        losses = super(SegmentationLosses, self).forward(seg_logit, seg_label)
        seg_label = seg_label.unsqueeze(1)
        seg_weight = self.sampler.sample(seg_logit, seg_label, losses)
        # apply weights and do the reduction, init is 'none'
        seg_weight = seg_weight.float()
        losses = weight_reduce_loss(
            losses, weight=seg_weight, reduction='mean', avg_factor=None)
        
        return losses

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
