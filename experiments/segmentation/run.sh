cd ../..
python setup.py install

cd experiments/segmentation


python train_sar.py --dataset sar_voc --child log_normal_new_noise_c1 --model deeplab --aux --backbone resnest269 \
--batch-size 30 --epochs 300 --warmup-epochs 2 
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 100 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 400 --warmup-epochs 2 --use-pretrain --frozen-stages -1
python test_sar.py --dataset sar_voc --child log_normal_new_noise_c1 --model deeplab --aux --backbone resnest269 \
--resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval

# # lr 0.0001
# # imagenet pretrain   warm 2 (fz -1)      with noise
# # 30    0.3710 2      0.4230 27           0.4255 30
# # 100                 0.5390 97
# # 200                 0.5792 184
# # 300                 0.5905 284

# pixAcc: 0.7402, mIoU: 0.5381, fwIoU: 0.5905: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:45<00:00,  2.66s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:35<00:29,  3.29s/it]
# IoU 0: 0.951237, IoU 1: 0.710608, IoU 2: 0.465442, IoU 3: 0.558243, IoU 4: 0.634782, IoU 5: 0.005291, IoU 6: 0.441033

# adam

# circle loss, focal loss

# cp ~/qiaohong/SAR/PyTorch-Encoding/experiments/segmentation/runs/sar_voc/deeplab/resnest269/best/model_best.pth.tar ~/qiaohong/
# python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume make_docker/model_best.pth.tar --eval

# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 300 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/checkpoint.pth.tar