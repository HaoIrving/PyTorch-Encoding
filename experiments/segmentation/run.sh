cd ../..
python setup.py install

cd experiments/segmentation

python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --batch-size 30 --epochs 20 --warmup-epochs 1

python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval

# lr 0.0001
        # no warm       warm 1  warm 2
# 10    0.3237 10   
# 15    0.3430 15       0.3445  0.3469
# 20    0.3533 20       

# 15 warm    
# 20 warm       1      
# 30 warm       1      
# 40 warm       2   0.4546    


# adam

# circle loss, focal loss
 