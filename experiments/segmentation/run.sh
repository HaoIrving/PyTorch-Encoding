cd ../..
python setup.py install

cd experiments/segmentation

python train_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --batch-size 30 --epochs 20

python test_sar.py --dataset sar_voc --model deeplab --aux --backbone resnest269 --resume runs/sar_voc/deeplab/resnest269/default/checkpoint.pth.tar --eval

# 15 no warm 0.5315
# 15 warm    
# 20 no warm 0.5286
# 20 warm    
