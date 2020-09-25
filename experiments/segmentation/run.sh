cd ../..
python setup.py install

cd experiments/segmentation

python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --batch-size 30 --epochs 10 --warmup-epochs 0

python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval

# lr 0.0001

# 10 no warm           
# 15 no warm           
# 15 warm    
# 20 no warm 
# 20 warm       1      
# 30 warm       1      
# 40 warm       2      


# adam

# circle loss, focal loss
 