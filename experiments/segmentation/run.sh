cd ../..
python setup.py install

cd experiments/segmentation

python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
--batch-size 30 --epochs 100 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 400 --warmup-epochs 2 --use-pretrain --frozen-stages -1
python test_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
--resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval

# # lr 0.0001
# # imagenet pretrain   warm 2 (fz -1)
# # 30    0.3710 2      0.4230 27
# # 100                 0.5390 97
# # 200                 0.5792 184
# # 300                 

# pixAcc: 0.7312, mIoU: 0.5274, fwIoU: 0.5792
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017 
# IoU 0: 0.950491, IoU 1: 0.708188, IoU 2: 0.450719, IoU 3: 0.543161, IoU 4: 0.626394, IoU 5: 0.000000, IoU 6: 0.412857


# adam

# circle loss, focal loss

# cp ~/qiaohong/SAR/PyTorch-Encoding/experiments/segmentation/runs/sar_voc/deeplab/resnest269/best/model_best.pth.tar ~/qiaohong/
# python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume make_docker/model_best.pth.tar --eval

python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
--batch-size 30 --epochs 300 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/checkpoint.pth.tar