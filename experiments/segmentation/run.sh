cd ../..
python setup.py install

cd experiments/segmentation

python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --batch-size 30 --epochs 400 --warmup-epochs 2 --use-pretrain --frozen-stages -1
python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval
# # lr 0.0001
# # imagenet pretrain   warm 2 (fz -1)
#         # no warm      
# # 30    0.3710 2
# # 40                  0.4546
# # 500                 0.5598 402

# pixAcc: 0.7011, mIoU: 0.4665, fwIoU: 0.5598
# freq0: 0.068049, freq1: 0.045627, freq2: 0.298004, freq3: 0.492258, freq4: 0.050610, freq5: 0.018766, freq6: 0.026686
# IoU 0: 0.950798, IoU 1: 0.569489, IoU 2: 0.395693, IoU 3: 0.665571, IoU 4: 0.284397, IoU 5: 0.192148, IoU 6: 0.207530

# # ade20k pretrain     warm 2 (fz -1)       warm 2 & fz 4
# # 30                  0.4039              0.3831 26

# adam

# circle loss, focal loss
