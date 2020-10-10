import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import pickle
import cv2

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

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train', indir=None, denoise=False, 
                 keep10_org3=False, child3="log_normal_new_noise_c1",
                 mode=None, child='log_normal_c3', transform=None, target_transform=None, **kwargs):
        super(VOCSegmentation_sar, self).__init__(root, split, mode, transform,
                                                  target_transform, **kwargs)
        self.keep10_org3 = keep10_org3
        if not keep10_org3:
            _base_dir = os.path.join('VOCdevkit_sar', child)
            _voc_root = os.path.join(self.root, _base_dir)
            _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
            _image_dir = os.path.join(_voc_root, 'JPEGImages')
            # train/val/test splits are pre-cut
            _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        else:
            _base_dir10 = os.path.join('VOCdevkit_sar', child)
            _voc_root10 = os.path.join(self.root, _base_dir10)
            _mask_dir = os.path.join(_voc_root10, 'SegmentationClass')
            _image_dir10 = os.path.join(_voc_root10, 'JPEGImages')
            # train/val/test splits are pre-cut
            _splits_dir = os.path.join(_voc_root10, 'ImageSets/Segmentation')

            _base_dir3 = os.path.join('VOCdevkit_sar', child3)
            _voc_root3 = os.path.join(self.root, _base_dir3)
            _image_dir3 = os.path.join(_voc_root3, 'JPEGImages')

        if self.mode == 'train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
        elif self.mode == 'val':  
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.mode == 'testval':  # set keep10_org3=True only when args.eval=True
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.mode == 'test': # set args.eval=True when keep10_org3, not support keep10_org3
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.mode == 'docker':
            lines = [f for f in os.listdir(indir)]
            lines.sort()
        else:
            raise RuntimeError('Unknown dataset split.')
        
        if not keep10_org3:
            self.images = []
        else:
            self.images10 = []
            self.images3 = []
        self.masks = []
        self.denoise = denoise
        self.HH_paths = []
        HH_root = "/home/sun/contest/data_preprocessed/HH"

        if self.mode != 'docker':
            with open(os.path.join(_split_f), "r") as lines:
                for line in tqdm(lines):
                    if not keep10_org3:
                        _image = os.path.join(_image_dir, line.rstrip('\n') + ".pkl")
                        assert os.path.isfile(_image)
                        self.images.append(_image)
                        _HH = os.path.join(HH_root, str(int(line.rstrip('\n')) + 1) + "_HH.tiff")
                        if os.path.exists(HH_root):
                            self.HH_paths.append(_HH)
                        else:
                            self.HH_paths.append("")
                    else:
                        _image10 = os.path.join(_image_dir10, line.rstrip('\n') + ".pkl")
                        assert os.path.isfile(_image10)
                        self.images10.append(_image10)
                        _image3 = os.path.join(_image_dir3, line.rstrip('\n') + ".pkl")
                        assert os.path.isfile(_image3)
                        self.images3.append(_image3)
                    if self.mode != 'test':
                        _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".pkl")
                        assert os.path.isfile(_mask)
                        self.masks.append(_mask)
            if self.mode != 'test':
                if not keep10_org3:
                    assert (len(self.images) == len(self.masks))
                else:
                    assert (len(self.images10) == len(self.masks))
        if self.mode == 'docker':
            for line in tqdm(lines):
                if line.split('.')[-1] != 'tif':
                    continue
                _image = os.path.join(indir, line)
                assert os.path.isfile(_image)
                self.images.append(_image)

    def __getitem__(self, index):
        if self.mode != 'docker':
            if not self.keep10_org3:
                # img = Image.open(self.images[index]).convert('RGB')
                img_f = open(self.images[index], 'rb') 
                img = pickle.load(img_f) # 512,512,3
            else:
                img_f10 = open(self.images10[index], 'rb') 
                img10 = pickle.load(img_f10) # 512,512,3
                img_f3 = open(self.images3[index], 'rb') 
                img3 = pickle.load(img_f3) # 512,512,3

            if self.mode == 'test': 
                if self.transform is not None:
                    img = self.transform(img)
                return img, os.path.basename(self.images[index]), self.HH_paths[index]

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
                if not self.keep10_org3:
                    img = self.transform(img)
                    if img.type() == 'torch.DoubleTensor':
                        img = img.type(torch.FloatTensor)
                else:
                    img10 = self.transform(img10)
                    if img10.type() == 'torch.DoubleTensor':
                        img10 = img10.type(torch.FloatTensor)
                    img3 = self.transform(img3)
                    if img3.type() == 'torch.DoubleTensor':
                        img3 = img3.type(torch.FloatTensor)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if not self.keep10_org3:
                return img, target, self.HH_paths[index]
            else:
                return img10, img3, target, HH_path

        if self.mode == 'docker':
            img_paths = self.images[index * 4: index * 4 + 4]
            if self.split == 'keep10':
                img = self.keep4_4c4_2c2_10(img_paths) # 512 512 10
            if self.split == 'c2':
                img = self.combination_2(img_paths)
            if self.transform is not None:
                img = self.transform(img)
                if img.type() == 'torch.DoubleTensor':
                    img = img.type(torch.FloatTensor)
            # x_HH_tiff = os.path.basename(self.images[index * 4])
            
            return img, img_paths[0]

    def keep4_4c4_2c2_10(self, img_paths):
        """
        在svm, nn上效果最好的特征组合
        :param img_paths:
        :param image, 4, 512, 512, 已经过log和归一化处理
        :return:
        """
        HH, HV, VH, VV = self.cat_4(img_paths, th=2)

        channel_0 = HH
        channel_0 = channel_0[:, :, np.newaxis]
        channel_1 = HV
        channel_1 = channel_1[:, :, np.newaxis]
        channel_2 = VH
        channel_2 = channel_2[:, :, np.newaxis]
        channel_3 = VV
        channel_3 = channel_3[:, :, np.newaxis]

        tmp = np.sqrt(HH * HH + VV * VV)

        channel_4 = VH / HH
        channel_4 = channel_4[:, :, np.newaxis]
        channel_5 = HV / tmp
        channel_5 = channel_5[:, :, np.newaxis]
        channel_6 = VH / tmp
        channel_6 = channel_6[:, :, np.newaxis]
        channel_7 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
        channel_7 = channel_7[:, :, np.newaxis]

        tmp1 = np.abs(HV + VH)
        tmp2 = HH + VV

        channel_8 = np.sqrt(tmp1 * tmp2)
        channel_8 = channel_8[:, :, np.newaxis]
        channel_9 = np.sqrt(HV * VH)
        channel_9 = channel_9[:, :, np.newaxis]

        ret = np.concatenate((channel_0, channel_1, channel_2, channel_3, channel_4,
                              channel_5, channel_6, channel_7, channel_8, channel_9), axis=2)
        return ret

    def combination_1(self, img_paths):
        HH, HV, VH, VV = self.cat_4(img_paths, th=2)
        if self.denoise:
            HH = HH / 10.553297276390468439899450459052
            HV = HV / 10.553297276390468439899450459052
            VH = VH / 10.553297276390468439899450459052
            VV = VV / 10.553297276390468439899450459052
        tmp = np.sqrt(HH * HH + VV * VV)
        channel_0 = VH / HH
        channel_0 = channel_0[:, :, np.newaxis]
        channel_1 = HV / tmp
        channel_1 = channel_1[:, :, np.newaxis]
        channel_2 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
        channel_2 = channel_2[:, :, np.newaxis]
        ret = np.concatenate((channel_0, channel_1, channel_2), axis=2)
        return ret

    def combination_2(self, img_paths):
        HH, HV, VH, VV = self.cat_4(img_paths, th=2)
        if self.denoise:
            HH = HH / 10.553297276390468439899450459052
            HV = HV / 10.553297276390468439899450459052
            VH = VH / 10.553297276390468439899450459052
            VV = VV / 10.553297276390468439899450459052
        tmp1 = np.abs(HV + VH)
        tmp2 = HH + VV
        channel_0 = np.sqrt(tmp1 * tmp2)
        channel_0 = channel_0[:, :, np.newaxis]
        channel_1 = np.sqrt(HV * VH)
        channel_1 = channel_1[:, :, np.newaxis]
        channel_2 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
        channel_2 = channel_2[:, :, np.newaxis]
        ret = np.concatenate((channel_0, channel_1, channel_2), axis=2)
        return ret

    def cat_4(self, img_paths, th=2):
        im_datas1 = self.tiff_np_log(img_paths[0], th)
        im_datas2 = self.tiff_np_log(img_paths[1], th)
        im_datas3 = self.tiff_np_log(img_paths[2], th)
        im_datas4 = self.tiff_np_log(img_paths[3], th)
        return im_datas1, im_datas2, im_datas3, im_datas4

    def tiff_np_log(self, img_path, th=2):
        im_datas_org = cv2.imread(img_path, -1)
        im_datas_org = np.clip(im_datas_org, th, None)
        im_datas = np.log(im_datas_org)
        im_datas = im_datas / 11.090339660644531250
        if self.denoise:
            im_datas = denoise(im_datas, im_datas_org)
        return im_datas

    def _mask_transform(self, mask):
        # return torch.from_numpy(np.array(mask)).long()
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        if self.mode != 'docker':
            if not self.keep10_org3:
                return len(self.images)
            else:
                return len(self.images10)
        if self.mode == 'docker':
            return len(self.images) // 4

    @property
    def pred_offset(self):
        return 0


def denoise(y, f, lambda_=1.3):
    p0x = np.zeros(y.shape)
    p0y = np.zeros(y.shape)
    x = y
    x0bar = x
    x0 = x
    rho = 2
    tol = 1e-5
    xr = []
    for i in range(1, 301):
        Mx, My = gradxy(x0bar)
        p1x = p0x + 1.0 / (2.0 * rho) * Mx
        p1y = p0y + 1.0 / (2.0 * rho) * My
        tmp = np.sqrt(p1x ** 2 + p1y ** 2)
        tmp = np.clip(tmp, 1, None)
        p1x = p1x / tmp
        p1y = p1y / tmp
        # Newton Method
        g = lambda_ * (-1.0 * f * f / (np.spacing(1) + np.exp(x)) / (np.spacing(1) + np.exp(x)) + 1) + 2 * rho * (x - x0) - div(p1x, p1y)
        gp = lambda_ * 1.0 * f * f / (np.spacing(1) + np.exp(x)) / (np.spacing(1) + np.exp(x)) + 2 * rho
        x = x - g / gp
        xr.append(np.linalg.norm(x - x0, ord='fro') / np.linalg.norm(x0, ord='fro'))
        if i > 1 and xr[-1] < tol:
            break
        # x1bar
        x1bar = 2 * x - x0
        x0 = x
        x0bar = x1bar
        p0x = p1x
        p0y = p1y
    return x

def div(px, py):
    m, n = px.shape
    Mx = np.zeros_like(px)
    My = np.zeros_like(px)

    Mx[1: m - 1, :] = px[1: m - 1, :] - px[: m - 2, :]
    Mx[0, :] = px[0, :]
    Mx[m - 1, :] = - px[m - 2, :]

    My[:, 1: n - 1] = py[:, 1: n - 1] - py[:, : n - 2]
    My[:, 0] = py[:, 0]
    My[:, n - 1] = - py[:, n - 2]
    return Mx + My

def gradxy(I):
    m, n = I.shape
    Mx = np.zeros_like(I)
    Mx[0: m - 1, :] = - I[0: m - 1, :] + I[1:, :]
    Mx[m - 1, :] = np.zeros(n)

    My = np.zeros_like(I)
    My[:, 0: n - 1] = - I[:, 0: n - 1] + I[:, 1:]
    My[:, n - 1] = np.zeros(m)
    return Mx, My