import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import pickle

import torch
from .base import BaseDataset


class VOCSegmentation_sar(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 7

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, child='log_normal_c3', transform=None, target_transform=None, **kwargs):
        super(VOCSegmentation_sar, self).__init__(root, split, mode, transform,
                                                  target_transform, **kwargs)
        _base_dir = os.path.join('VOCdevkit_sar', child)
        _voc_root = os.path.join(self.root, _base_dir)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if self.mode == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif self.mode == 'val':  
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.mode == 'testval':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".pkl")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".pkl")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        # img = Image.open(self.images[index]).convert('RGB')
        img_f = open(self.images[index], 'rb') 
        img_np = pickle.load(img_f)
        img = img_np.transpose(1, 2, 0)  # 512,512,3
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # target = Image.open(self.masks[index])
        mask_f = open(self.masks[index], 'rb')
        target_np = pickle.load(mask_f)  # 512,512
        target = Image.fromarray(target_np.astype('uint8'))
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform_sar(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform_sar(img, target)
        else:
            assert self.mode == 'testval'
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _mask_transform(self, mask):
        # return torch.from_numpy(np.array(mask)).long()
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0
