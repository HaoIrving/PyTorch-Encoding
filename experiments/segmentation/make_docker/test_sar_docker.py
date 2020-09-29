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
    args.model = "deeplab"
    args.aux = True
    args.backbone = "resnest269"
    args.resume = "model_best.pth.tar"
    # args.eval = True
    args.docker = True
    args.c1 = True
    # args.c2 = True

    # folder
    indir = "/input_path"
    outdir = '/output_path'
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
    # model
    pretrained = args.resume is None and args.verify is None
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=pretrained)
        model.base_size = args.base_size
        model.crop_size = args.crop_size
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset, root='/encoding/models',
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
    if args.acc_bn:
        from encoding.utils.precise_bn import update_bn_stats
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        trainloader = data.DataLoader(ReturnFirstClosure(trainset), batch_size=args.batch_size,
                                      drop_last=True, shuffle=True, **loader_kwargs)
        print('Reseting BN statistics')
        #model.apply(reset_bn_statistics)
        model.cuda()
        update_bn_stats(model, trainloader)

    if args.export:
        torch.save(model.state_dict(), args.export + '.pth')
        return

    # TODO: using multi scale testing
    scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
            []# [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]#, 2.0
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales).cuda()
    evaluator.eval()
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
                outputs = evaluator.parallel_forward(image)
                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                mask = utils.get_mask_pallete(predict, args.dataset)
                basename = os.path.splitext(impath)[0]
                basename = basename.split('_')[0]
                outname = basename + '_gt.png'
                mask.save(os.path.join(outdir, outname))
                get_xml(basename)

    if args.eval:
        print('freq0: %f, freq1: %f, freq2: %f, freq3: %f, freq4: %f, freq5: %f, freq6: %f' % \
            (freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6]))
        print('IoU 0: %f, IoU 1: %f, IoU 2: %f, IoU 3: %f, IoU 4: %f, IoU 5: %f, IoU 6: %f' % \
            (IoU[0], IoU[1], IoU[2], IoU[3], IoU[4], IoU[5], IoU[6] ))

def get_xml(filename):
    root=ET.Element('annotation')
    root.text='\n'
    tree=ET.ElementTree(root)

    #parameters to set
    #filename=os.walk('/input_path')[2]
    filename=filename
    resultfile=filename.split('.')[0]+'_gt.png'
    resultfile_xml='./'+filename.split('.')[0]+'.xml'
    organization='CASIA'
    author='1,2,3,4,5,6'

    element_source=ET.Element('source')
    element_source.text='\n'+7*' '
    element_source.tail='\n'+4*' '
    element_filename=ET.Element('filename')
    element_filename.tail='\n'+7*' '
    element_filename.text=filename

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
