###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2

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
        parser.add_argument('--docker', action='store_true', default= False,
                            help='generate masks on val set')
        parser.add_argument('--c1', action='store_true', default= False,
                            help='generate masks on val set')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

def test(args):
    # 
    args.dataset = "sar_voc" 
    # args.model = "deeplab"
    args.aux = True
    args.backbone = "resnest269"
    # args.resume = "model_best_noise_6272.pth.tar"
    # args.eval = True
    args.docker = True
    args.c1 = True
    # args.c2 = True
    args.workers = 0


    # folder
    indir = "experiments/segmentation/make_docker/input_path"
    # indir = "./input_path"
    outdir = 'experiments/segmentation/make_docker//output_path'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        # transform.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
    # dataset
    if args.eval:
        testset = get_dataset(args.dataset, split='val', mode='testval',
                              transform=input_transform)
    elif args.test_val:
        testset = get_dataset(args.dataset, split='val', mode='test',
                              transform=input_transform)
    elif args.docker and args.c1:
        testset = get_dataset(args.dataset, split='c1', mode='docker', indir=indir, 
                              transform=input_transform)
    elif args.docker and args.c2:
        testset = get_dataset(args.dataset, split='c2', mode='docker', indir=indir, 
                              transform=input_transform)
    else:
        testset = get_dataset(args.dataset, split='test', mode='test',
                              transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    
    # MODEL ASSEMBLE
    resume = [
        "experiments/segmentation/make_docker/psp_noise_6596.pth.tar",
        # "experiments/segmentation/make_docker/deeplab_noise_6272.pth.tar", 
        # "experiments/segmentation/make_docker/encnet_noise_6190.pth.tar", 
        ]
    ioukeys = [path.split("/")[-1].split(".")[0] for path in resume]
    ioutable = {
        "psp_noise_6596":      [0.944471, 0.736214, 0.639560, 0.608305, 0.669817, 0.302915, 0.685308],
        "psp_noise_6549":      [0.943818, 0.737374, 0.635447, 0.605156, 0.665047, 0.288825, 0.632505],
        "deeplab_noise_6272":  [0.959670, 0.673592, 0.538992, 0.611297, 0.660384, 0.185302, 0.339777],
        "encnet_noise_6190":   [0.966603, 0.679119, 0.530339, 0.601795, 0.656715, 0.097908, 0.265538],
        "psp_noise_6122":      [0.952692, 0.685647, 0.523152, 0.601464, 0.631610, 0.060363, 0.389081],
        "deeplab_noise_5999":  [0.947047, 0.646243, 0.508729, 0.583762, 0.641149, 0.041550, 0.274465],
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
        # model
        pretrained = args.resume is None and args.verify is None
        model = get_segmentation_model(args.model, dataset=args.dataset, root='.',
                                    backbone=args.backbone, aux = args.aux,
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
        # print(model)
        evaluator = MultiEvalModule(model, testset.num_class, scales=scales).cuda()
        evaluator.eval()
        evaluators[i] = evaluator

    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        if args.eval:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                # outputs = model(image)
                # outputs = gather(outputs, 0, dim=0)
                # pred = outputs[0]
                metric.update(dst, predicts)
                pixAcc, mIoU, fwIoU, freq, IoU = metric.get()
                tbar.set_description('pixAcc: %.4f, mIoU: %.4f, fwIoU: %.4f' % (pixAcc, mIoU, fwIoU))
        else:
            with torch.no_grad():
                # model_assemble
                predicts = []
                for i in range(assemble_nums):
                    predict = evaluators[i].parallel_forward(image) # [tensor([1, 7, 512, 512], cuda0), tensor]
                    predicts.append(predict)
                
                weighted_asmb = True
                if weighted_asmb:
                    outputs = model_assemble(predicts, weights, assemble_nums)

                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                # predict = postprocess(predict)
                mask = utils.get_mask_pallete(predict, args.dataset)
                # basename = basename.split('_')[0]
                d = ["HH", "HV", "VH", "VV"]
                paths = impath.split(" ")
                for i in range(4):
                    basename = os.path.splitext(paths[i])[0]
                    outname = basename + '_gt.png'
                    mask.save(os.path.join(outdir, outname))
                    get_xml(outdir, paths[i], basename, outname)

    if args.eval:
        print('freq0: %f, freq1: %f, freq2: %f, freq3: %f, freq4: %f, freq5: %f, freq6: %f' % \
            (freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6]))
        print('IoU 0: %f, IoU 1: %f, IoU 2: %f, IoU 3: %f, IoU 4: %f, IoU 5: %f, IoU 6: %f' % \
            (IoU[0], IoU[1], IoU[2], IoU[3], IoU[4], IoU[5], IoU[6] ))

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

def get_xml(outdir, filename, basename, outname):
    root=ET.Element('annotation')
    root.text='\n'
    tree=ET.ElementTree(root)

    #parameters to set
    #filename=os.walk('/input_path')[2]
    filename = filename
    resultfile = outname
    resultfile_xml = basename + '.xml'
    resultfile_xml = os.path.join(outdir, resultfile_xml)
    organization='CASIA'
    author='1,2,3,4,5,6'

    element_source=ET.Element('source')
    element_source.text='\n'+7*' '
    element_source.tail='\n'+4*' '
    element_filename=ET.Element('filename')
    element_filename.tail='\n'+7*' '
    element_filename.text= filename

    element_origin=ET.Element('origin')
    element_origin.tail='\n'+4*' '
    element_origin.text='GF2/GF3'
    element_research=ET.Element('research')
    element_research.text='\n'+7*' '
    element_research.tail='\n'+4*' '
    element_version=ET.Element('version')
    element_version.tail='\n'+7*' '
    element_version.text='4.0'
    element_provider=ET.Element('provider')
    element_provider.tail='\n'+7*' '
    element_provider.text=organization
    element_author=ET.Element('author')
    element_author.text=author
    element_author.tail='\n'+7*' '
    element_pluginname=ET.Element('pluginname')
    element_pluginname.tail='\n'+7*' '
    element_pluginname.text='地物标注'
    element_pluginclass=ET.Element('pluginclass')
    element_pluginclass.tail='\n'+7*' '
    element_pluginclass.text='标注'
    element_time=ET.Element('time')
    element_time.tail='\n'+4*' '
    element_time.text='2020-07-2020-11'
    element_seg=ET.Element('segmentation')
    element_seg.text='\n'+7*' '
    element_seg.tail='\n'
    element_resultfile=ET.Element('resultflie')
    element_resultfile.tail='\n'+4*' '
    element_resultfile.text=resultfile

    #add
    element_source.append(element_filename)
    element_source.append(element_origin)

    element_research.append(element_version)
    element_research.append(element_provider)
    element_research.append(element_author)
    element_research.append(element_pluginname)
    element_research.append(element_pluginclass)
    element_research.append(element_time)

    element_seg.append(element_resultfile)

    root.append(element_source)
    root.append(element_research)
    root.append(element_seg)
    #write
    tree.write(resultfile_xml,encoding='utf-8',xml_declaration=True)

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