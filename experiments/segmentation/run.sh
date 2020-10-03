cd ../..
python setup.py install

cd experiments/segmentation


python train_sar.py --dataset sar_voc --child log_normal_new_noise_c1 --model encnet --aux --backbone resnest269 \
--batch-size 30 --epochs 400 --warmup-epochs 2 
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 100 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar
# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 400 --warmup-epochs 2 --use-pretrain --frozen-stages -1
python test_sar.py --dataset sar_voc --child log_normal_new_noise_c1 --model encnet --aux --backbone resnest269 \
--resume runs/sar_voc/encnet/resnest269/default/model_best.pth.tar --eval

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
# 400                   

# adam

# circle loss, focal loss

# cp ~/qiaohong/SAR/PyTorch-Encoding/experiments/segmentation/runs/sar_voc/deeplab/resnest269/best/model_best.pth.tar ~/qiaohong/
# python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume make_docker/model_best.pth.tar --eval

# python train_sar.py --dataset sar_voc --child log_normal_new_c1 --model deeplab --aux --backbone resnest269 \
# --batch-size 30 --epochs 300 --warmup-epochs 2 --resume runs/sar_voc/deeplab/resnest269/default/checkpoint.pth.tar

# The ratio of class 0 is (0.058547342463991770833420247300)
# The ratio of class 1 is (0.086003739926268862880931465043)
# The ratio of class 2 is (0.180170923996913590059421039768)
# The ratio of class 3 is (0.362124059177812096521620333078)
# The ratio of class 4 is (0.276800937607167341081293443494)
# The ratio of class 5 is (0.015108341049382715756688178033)
# The ratio of class 6 is (0.021244655778463648887477432936)

# The ratio of class 0 is (0.058466728515624996620481113041)
# The ratio of class 1 is (0.085990172526041672163010787244)
# The ratio of class 2 is (0.180229236111111107421223209712)
# The ratio of class 3 is (0.362021213107638906159024827502)
# The ratio of class 4 is (0.276933541124131932242846687586)
# The ratio of class 5 is (0.015110536024305554897440018181)
# The ratio of class 6 is (0.021248572591145833965420308687)

# The ratio of class 0 is (0.058466728515624996620481113041)
# The ratio of class 1 is (0.085990172526041672163010787244)
# The ratio of class 2 is (0.180229236111111107421223209712)
# The ratio of class 3 is (0.362021213107638906159024827502)
# The ratio of class 4 is (0.276933541124131932242846687586)
# The ratio of class 5 is (0.015110536024305554897440018181)
# The ratio of class 6 is (0.021248572591145833965420308687)

# The ratio of class 0 is (0.055191559765625002043076818836)
# The ratio of class 1 is (0.085369403320312495231370064630)
# The ratio of class 2 is (0.183500315917968753742783860616)
# The ratio of class 3 is (0.369634570751953106526599412973)
# The ratio of class 4 is (0.271032675097656272633628304902)
# The ratio of class 5 is (0.014225178271484374692423813258)
# The ratio of class 6 is (0.021046296874999998599564676738)