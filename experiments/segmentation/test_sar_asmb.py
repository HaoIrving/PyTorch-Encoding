###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import cv2
# import copy
from scipy import stats


import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=512,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--acc-bn', action='store_true', default= False,
                            help='Re-accumulate BN statistics')
        parser.add_argument('--test-val', action='store_true', default= False,
                            help='generate masks on val set')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # 
        parser.add_argument('--child', type=str, default='log_normal_new_c1',
                            help='dataset name (default: pascal12)')                            
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

def test(args):
    args.dataset = "sar_voc"
    args.child = "log_normal_new_noise_4channel_keep4_4c4_2c2_10_blc700"
    args.aux = True
    args.backbone = "resnest269"

    args.workers = 0

    # args.eval = True
    keep10_org3 = True

    # args.model = "deeplab"
    # args.resume = "experiments/segmentation/make_docker/model_best_noise_6272.pth.tar"

    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        # transform.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
    # dataset
    if args.eval:
        testset = get_dataset(args.dataset, split='val', mode='testval', child=args.child, 
                              keep10_org3=keep10_org3, child3="log_normal_new_noise_c1",
                              transform=input_transform)
    elif args.test_val:
        testset = get_dataset(args.dataset, split='val', mode='test',
                              transform=input_transform)
    else:
        testset = get_dataset(args.dataset, split='val', mode='test', child=args.child, 
                              keep10_org3=keep10_org3,
                              transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)

    # MODEL ASSEMBLE
    """
    001 7236
    01 7231
    015 7209
    014 7220
    """
    resume = [
        "experiments/segmentation/make_docker/psp_noise_7123_keep10.pth.tar",
        "experiments/segmentation/make_docker/upernet_noise_7096_keep10.pth.tar",
        # "experiments/segmentation/make_docker/psp_noise_6596.pth.tar",
        # "experiments/segmentation/make_docker/psp_noise_6549.pth.tar",
        # "experiments/segmentation/make_docker/psp_noise_6450_keep10.pth.tar",
        # "experiments/segmentation/make_docker/deeplab_noise_6272.pth.tar", 
        # "experiments/segmentation/make_docker/encnet_noise_6190.pth.tar", 
        # "experiments/segmentation/make_docker/psp_noise_6122.pth.tar",
        # "experiments/segmentation/make_docker/fcfpn_noise_6034_keep10.pth.tar",
        # "experiments/segmentation/make_docker/deeplab_noise_5999.pth.tar", 
        ]

    ioukeys = [path.split("/")[-1].split(".")[0] for path in resume]
    ioutable = {
        "psp_noise_7123_keep10":    [0.917917, 0.796020, 0.703411, 0.680619, 0.708302, 0.345198, 0.735613],
        "upernet_noise_7096_keep10":[0.929088, 0.788786, 0.689905, 0.681493, 0.704899, 0.337708, 0.742923],
        "psp_noise_6596":           [0.944471, 0.736214, 0.639560, 0.608305, 0.669817, 0.302915, 0.685308],
        "psp_noise_6549":           [0.943818, 0.737374, 0.635447, 0.605156, 0.665047, 0.288825, 0.632505],
        "deeplab_noise_6272":       [0.959670, 0.673592, 0.538992, 0.611297, 0.660384, 0.185302, 0.339777],
        "encnet_noise_6190":        [0.966603, 0.679119, 0.530339, 0.601795, 0.656715, 0.097908, 0.265538],
        "psp_noise_6122":           [0.952692, 0.685647, 0.523152, 0.601464, 0.631610, 0.060363, 0.389081],
        "deeplab_noise_5999":       [0.947047, 0.646243, 0.508729, 0.583762, 0.641149, 0.041550, 0.274465],
    }
    assemble_nums = len(resume)
    scales = []
    evaluators = defaultdict()
    weights = []
    for i in range(assemble_nums):
        ioukey  = ioukeys[i]
        iou     = ioutable[ioukey] # [0.959670, 0.673592, 0.538992, 0.611297, 0.660384, 0.185302, 0.339777]
        weights.append(iou)
    weights = np.array(weights)
    weights = weights / weights.sum(0) # restrict to [0, 1]
    weights = torch.from_numpy(weights).float()

    for i in range(assemble_nums):
        args.resume = resume[i]
        modelname = args.resume.split("/")[-1]
        args.model  = modelname.split("_")[0]
        if args.model == "upernet" or args.model == "fcfpn":
            args.aux = False
        if args.model == "encnet" or args.model == "psp" or args.model == "deeplab":
            args.aux = True
        
        keep10 = modelname.split(".")[0].split("_")[-1] == "keep10"
        
        # model
        pretrained = args.resume is None and args.verify is None
        model = get_segmentation_model(args.model, dataset=args.dataset, keep10=keep10,
                                    backbone=args.backbone, aux=args.aux,
                                    se_loss=args.se_loss,
                                    norm_layer=torch.nn.BatchNorm2d if args.acc_bn else SyncBatchNorm,
                                    base_size=args.base_size, crop_size=args.crop_size)
        # resuming checkpoint
        if args.verify is not None and os.path.isfile(args.verify):
            print("=> loading checkpoint '{}'".format(args.verify))
            model.load_state_dict(torch.load(args.verify))
        elif args.resume is not None and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            # strict=False, so that it is compatible with old pytorch saved models
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif not pretrained:
            raise RuntimeError ("=> no checkpoint found")

        evaluator = MultiEvalModule(model, testset.num_class, scales=scales, keep10=keep10).cuda()
        evaluator.eval()
        evaluators[i] = evaluator
    
    metric = utils.SegmentationMetric(testset.num_class)

    try:
        tbar = tqdm(test_data)#, ncols=10)
        for i, img10_img3_dst_HH_paths in enumerate(tbar):
            if not keep10_org3:
                image, dst, HH_paths = img10_img3_dst_HH_paths
            else:
                image10, image3, dst, HH_paths = img10_img3_dst_HH_paths
            if args.eval:
                with torch.no_grad():
                    # model_assemble
                    predicts = []
                    for i in range(assemble_nums):
                        if not keep10_org3:
                            predict = evaluators[i].parallel_forward(image) # [tensor([1, 7, 512, 512], cuda0), tensor]
                        else:
                            if evaluators[i].keep10 == True:
                                predict = evaluators[i].parallel_forward(image10)
                            if evaluators[i].keep10 == False:
                                predict = evaluators[i].parallel_forward(image3)
                        predicts.append(predict)
                    
                    weighted_asmb = True
                    if weighted_asmb:
                        predicts = model_assemble(predicts, weights, assemble_nums)

                    metric.update(dst, predicts, HH_paths, weighted_asmb=weighted_asmb, postproc=False)
                    pixAcc, mIoU, fwIoU, freq, IoU, confusion_matrix = metric.get()
                    tbar.set_description('pixAcc: %.4f, mIoU: %.4f, fwIoU: %.4f' % (pixAcc, mIoU, fwIoU))
            else:
                with torch.no_grad():
                    # model_assemble
                    weighted_asmb = False
                    if weighted_asmb:
                        predicts = []
                        for i in range(assemble_nums):
                            if not keep10_org3:
                                outputs = evaluators[i].parallel_forward(image) # [tensor([1, 7, 512, 512], cuda0), tensor]
                            else:
                                if evaluators[i].keep10 == True:
                                    outputs = evaluators[i].parallel_forward(image10)
                                if evaluators[i].keep10 == False:
                                    outputs = evaluators[i].parallel_forward(image3)
                            predicts.append(outputs)
                        outputs = model_assemble(predicts, weights, assemble_nums)
                        predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                                    for output in outputs]
                    
                    vote = True
                    if vote:
                        all_predicts = []
                        for i in range(assemble_nums):
                            if not keep10_org3:
                                outputs = evaluators[i].parallel_forward(image)
                            if keep10_org3:
                                if evaluators[i].keep10 == True:
                                    outputs = evaluators[i].parallel_forward(image10)
                                if evaluators[i].keep10 == False:
                                    outputs = evaluators[i].parallel_forward(image3)
                            # [tensor([1, 7, 512, 512], cuda0), tensor]
                            predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy().squeeze()) 
                                    for output in outputs]
                            all_predicts.append(predicts)
                        predicts = model_vote(all_predicts, assemble_nums)

                for predict, impath, HH_path in zip(predicts, dst, HH_paths):
                    # predict = postprocess(predict)
                    # if 0 in predict and HH_path != "": 
                    #     HH = cv2.imread(HH_path, -1)
                    #     predict = black_area(HH, predict[0])

                    mask = utils.get_mask_pallete(predict, args.dataset)
                    mask_gray = Image.fromarray(predict.squeeze().astype('uint8')) #
                    basename = os.path.splitext(impath)[0]
                    basename = basename.split('_')[0]
                    basename = str(int(basename) + 1)
                    outname = basename + '_visualize.png'#
                    outname_gray = basename + '_feature.png'#
                    mask.save(os.path.join(outdir, outname))
                    mask_gray.save(os.path.join(outdir, outname_gray))#
    except KeyboardInterrupt:
        tbar.close()
        raise
    tbar.close()

    if args.eval:
        print('freq0: %f, freq1: %f, freq2: %f, freq3: %f, freq4: %f, freq5: %f, freq6: %f' % \
            (freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6]))
        print('IoU 0: %f, IoU 1: %f, IoU 2: %f, IoU 3: %f, IoU 4: %f, IoU 5: %f, IoU 6: %f' % \
            (IoU[0], IoU[1], IoU[2], IoU[3], IoU[4], IoU[5], IoU[6] ))

def model_vote(predicts, n):
    ngpus = len(predicts[0])
    voted = []
    for gpu_index in range(ngpus):
        height, width = predicts[0][gpu_index].shape
        vote_mask = np.zeros((height, width))
        for h in range(height):
            for w in range(width):
                record = np.zeros((1,7))
                for model_index in range(n):  # n models
                    mask = predicts[model_index][gpu_index]
                    pixel = mask[h,w]
                    record[0,pixel] += 1
                label = record.argmax()
                vote_mask[h,w] = label
        voted.append(vote_mask)
    return voted

def black_area(im_data1, pre_lab):
    # load  the  corresponding  original    data.    use    two    channel is enough
    # a1 = im_data1  # HH channel
    # a4 = im_data2  # VV channel
    b = np.where(im_data1 == 0)  # find the index where a1==0 and a4==0
    b0 = b[0].shape
    c = np.ones((512, 512))
    z = pre_lab

    for i in range(0, b0[0]):
        c[b[0][i]][b[1][i]] = 0  # 找出原图中黑色区域，做成mask c，c中0元素对应的为黑色，其它为1
        z[b[0][i]][b[1][i]] = 0  # z为最终要输出的图，先将确定的黑色区域变成0

    d = np.where(pre_lab == 0)  # 找出预测图为0的位置
    d0 = d[0].shape
    
    for i in range(0, d0[0]):
        if (c[d[0][i]][d[1][i]]) == 1:
            q = []
            hw = 40
            while q == []:
                f0 = np.lib.pad(pre_lab, ((hw, hw), (hw, hw)), 'constant')  # 将图像做对称扩展
                p = f0[d[0][i]:d[0][i] + 2 * hw - 1, d[1][i]:d[1][i] + 2 * hw - 1]  # 取出该点对应的patch
                e = np.nonzero(p)  # 将其label中 非零元素的众数作为该点的label
                e0 = e[0].shape
                for j in range(0, e0[0]):
                    q.append(p[e[0][j]][e[1][j]])
                if q == []:
                    hw += 40
            z[d[0][i]][d[1][i]] = stats.mode(q)[0][0]
    return z

def postprocess(predict):
    """both
    18 0.6575
    17 0.6583
    16 0.6576
    """
    ret = np.zeros_like(predict)
    for i in range(predict.shape[0]):
        img = predict[i].astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))# 正方形 8*8
        # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        ret[i] = closing
    return ret

def model_assemble(predicts, weights, n_models):
    """
    predicts: list(list(tensor))
    weights: np array, 3 * 7
    array([[0.32884602, 0.35279618, 0.37274472, 0.33282369, 0.33551868, 0.50490792, 0.51098302],
           [0.33436919, 0.32227972, 0.31616551, 0.33620111, 0.33316617, 0.32393472, 0.27449629],
           [0.33678479, 0.32492411, 0.31108977, 0.3309752 , 0.33131515, 0.17115736, 0.21452069]])
    """
    ngpus = len(predicts[0])
    assembled = [torch.zeros_like(predicts[0][i]) for i in range(ngpus)]
    for i in range(n_models):
        predict = predicts[i] # [tensor([1, 7, 512, 512], cuda0), tensor(cuda1)]
        for cuda_index, pred in enumerate(predict): # tensor([1, 7, 512, 512], cuda0)
            pred = pred.transpose(1, 2).transpose(2, 3).contiguous() # [1, 512, 512, 7]
            pred = pred * weights[i].cuda(pred.device) # [1, 512, 512, 7], [7]
            pred = pred.transpose(2, 3).transpose(1, 2).contiguous() # [1, 7, 512, 512]
            assembled[cuda_index] += pred
    return assembled

class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
