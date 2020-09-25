cd ../..
python setup.py install

cd experiments/segmentation

python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --batch-size 30 --epochs 40

python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume runs/sar_voc/deeplab/resnest269/default/model_best.pth.tar --eval

# 15 no warm            0.5315
# 15 warm    
# 20 no warm 
# 20 warm       1      0.5469 19
# 30 warm       1      0.52   23
# 40 warm       2      


# adam

# circle loss, focal loss
 