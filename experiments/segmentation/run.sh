cd ../..
python setup.py install

cd experiments/segmentation

# python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --batch-size 30 --epochs 30 --warmup-epochs 0
# python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval
# # lr 0.0001
#         # no warm       warm 1      warm 2
# # 20    0.3533 20       0.3574 20   
# # 30    
# # 40    
python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest50 --batch-size 30 --epochs 30 --warmup-epochs 0
python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest50 --resume runs/sar_voc/deeplab/resnest50/default/model_best.pth.tar --eval
# # lr 0.0001
#         # no warm       warm 1      warm 2
# # 20    
# # 30    
# # 40 


# adam

# circle loss, focal loss
 