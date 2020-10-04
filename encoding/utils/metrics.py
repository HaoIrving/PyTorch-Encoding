##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import threading
import numpy as np
import torch
import cv2

__all__ = ['accuracy', 'get_pixacc_miou',
           'SegmentationMetric', 'batch_intersection_union', 'batch_pix_accuracy',
           'pixel_accuracy', 'intersection_and_union']

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_pixacc_miou(total_correct, total_label, total_inter, total_union, total_lab):
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    freq = 1.0 * total_lab / (np.spacing(1) + total_label)
    fwIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
    
    return pixAcc, mIoU, fwIoU, freq, IoU #, total_lab
    

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds, weighted_asmb=True, postproc=True):
        def evaluate_worker(self, label, pred, weighted_asmb=True, postproc=True):
            correct, labeled = batch_pix_accuracy(
                pred, label, weighted_asmb=weighted_asmb, postproc=postproc)
            inter, union, area_lab = batch_intersection_union(
                pred, label, self.nclass, weighted_asmb=weighted_asmb, postproc=postproc)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
                self.total_lab += area_lab
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds, weighted_asmb=weighted_asmb, postproc=postproc)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred, weighted_asmb, postproc),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_correct, self.total_label, self.total_inter, self.total_union

    def get(self):
        return get_pixacc_miou(self.total_correct, self.total_label, self.total_inter, self.total_union, self.total_lab)
 
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_lab = 0
        return

def postprocess(predict):
    """both
    18 0.6575
    17 0.6583
    16 0.6576
    """
    ret = np.zeros_like(predict)
    for i in range(predict.shape[0]):
        img = predict[i].astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))# 正方形 8*8
        # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        ret[i] = closing
        # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
        # kernel = np.ones((18, 18), np.uint8)
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # ret[i] = opening

    return ret


def batch_pix_accuracy(output, target, weighted_asmb=False, postproc=False):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1) # 1, 512, 512

    predict = predict.cpu().numpy()
    if postproc:
        predict = postprocess(predict)
    predict = predict.astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, weighted_asmb=False, postproc=False):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass

    predict = predict.cpu().numpy()
    if postproc:
        predict = postprocess(predict)
    predict = predict.astype('int64')  + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union, area_lab


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    #pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image. 
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class-1,
                                        range=(1, num_class - 1))
    # Compute area union: 
    area_pred, _ = np.histogram(im_pred, bins=num_class-1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class-1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union
