cd ../..
python setup.py install

cd experiments/segmentation


python train_sar.py --dataset sar_voc --child log_normal_new_noise_c1 --model psp  --backbone resnest269 \
--batch-size 30 --epochs 30 --warmup-epochs 2  --aux  --ohem # --se-loss
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 100 --warmup-epochs 2 
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 400 --warmup-epochs 2 --use-pretrain --frozen-stages -1
python test_sar.py --dataset sar_voc --child log_normal_new_noise_c1 --model psp  --backbone resnest269 \
--resume runs/sar_voc/psp/resnest269/default/model_best.pth.tar --eval  --aux --se-loss
# 30                        ohem
# fcn       0.4654 25
# psp       0.5063 30       
#fcfpn      0.4734 30
# atten     0.4652 24
# encnet    0.4534 30 
# 

# deeplab
# # lr 0.0001
# # imagenet pretrain   warm 2 (fz -1)      with noise
# # 30    0.3710 2      0.4230 27           0.4255 30
# # 100                 0.5390 97
# # 200                 0.5792 184
# # 300                 0.5905 284          0.5999 293
# # 500                                     0.6272 330/475
# pixAcc: 0.7402, mIoU: 0.5381, fwIoU: 0.5905: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:45<00:00,  2.66s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:35<00:29,  3.29s/it]
# IoU 0: 0.951237, IoU 1: 0.710608, IoU 2: 0.465442, IoU 3: 0.558243, IoU 4: 0.634782, IoU 5: 0.005291, IoU 6: 0.441033
# pixAcc: 0.7489, mIoU: 0.5204, fwIoU: 0.5999: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:49<00:00,  2.92s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:38<00:32,  3.64s/it]
# IoU 0: 0.947047, IoU 1: 0.646243, IoU 2: 0.508729, IoU 3: 0.583762, IoU 4: 0.641149, IoU 5: 0.041550, IoU 6: 0.274465
# pixAcc: 0.7690, mIoU: 0.5670, fwIoU: 0.6272: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:44<00:00,  2.63s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:34<00:29,  3.22s/it]
# IoU 0: 0.959670, IoU 1: 0.673592, IoU 2: 0.538992, IoU 3: 0.611297, IoU 4: 0.660384, IoU 5: 0.185302, IoU 6: 0.339777

# encnet                wo se               w se
# 400                   0.6190 363
# 500 
# pixAcc: 0.7635, mIoU: 0.5426, fwIoU: 0.6190: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:45<00:00,  2.69s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:35<00:30,  3.35s/it]
# IoU 0: 0.966603, IoU 1: 0.679119, IoU 2: 0.530339, IoU 3: 0.601795, IoU 4: 0.656715, IoU 5: 0.097908, IoU 6: 0.265538 


# pspnet
# 400                   0.6122 138          0.6549 396
# 500                   0.6596 488
# pixAcc: 0.7583, mIoU: 0.5491, fwIoU: 0.6122: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:45<00:00,  2.65s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:35<00:29,  3.31s/it]
# IoU 0: 0.952692, IoU 1: 0.685647, IoU 2: 0.523152, IoU 3: 0.601464, IoU 4: 0.631610, IoU 5: 0.060363, IoU 6: 0.389081
# pixAcc: 0.7831, mIoU: 0.6440, fwIoU: 0.6549: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:44<00:00,  2.63s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:35<00:29,  3.30s/it]
# IoU 0: 0.943818, IoU 1: 0.737374, IoU 2: 0.635447, IoU 3: 0.605156, IoU 4: 0.665047, IoU 5: 0.288825, IoU 6: 0.632505
# pixAcc: 0.7877, mIoU: 0.6552, fwIoU: 0.6596: 100%|������������������������������������������������������������������������������������������������������| 17/17 [00:48<00:00,  2.84s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017                                    | 8/17 [00:37<00:31,  3.52s/it]
# IoU 0: 0.944471, IoU 1: 0.736214, IoU 2: 0.639560, IoU 3: 0.608305, IoU 4: 0.669817, IoU 5: 0.302915, IoU 6: 0.685308
# adam

# circle loss, focal loss

# cp ~/qiaohong/SAR/PyTorch-Encoding/experiments/segmentation/runs/sar_voc/deeplab/resnest269/best/model_best.pth.tar ~/qiaohong/
# python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume make_docker/model_best.pth.tar --eval

# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 300 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/checkpoint.pth.tar

# ASSEMBLE
# => loaded checkpoint 'experiments/segmentation/make_docker/psp_noise_6549.pth.tar' (epoch 396)
# MultiEvalModule: base_size 512, crop_size 480
# frozen_stages is -1
# => loaded checkpoint 'experiments/segmentation/make_docker/deeplab_noise_6272.pth.tar' (epoch 475)
# MultiEvalModule: base_size 512, crop_size 480
# frozen_stages is -1
# => loaded checkpoint 'experiments/segmentation/make_docker/encnet_noise_6190.pth.tar' (epoch 363)
# MultiEvalModule: base_size 512, crop_size 480
# pixAcc: 0.7981, mIoU: 0.6779, fwIoU: 0.6721: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:37<00:00,  1.58s/it]
# freq0: 0.061955, freq1: 0.078376, freq2: 0.163651, freq3: 0.357225, freq4: 0.298697, freq5: 0.016080, freq6: 0.024017
# IoU 0: 0.965633, IoU 1: 0.772948, IoU 2: 0.632190, IoU 3: 0.623202, IoU 4: 0.678333, IoU 5: 0.352633, IoU 6: 0.720387
