# pytorch-semantic-segmentation
## Reference
- *[ERFNet](https://github.com/Eromera/erfnet_pytorch)*
- *[PiWise](https://github.com/bodokaiser/piwise)*
## Network
- fcn
- segnet
- erfnet
- pspnet
- unet
## Environment
- pytorch 0.3.0
- torchvision 0.2.0
- python 3.5.2
## Train
 `CUDA_VISIBLE_DEVICES=0 python3 train.py --datadir ./data/train --savedir ./save_models/ --model segnet`  
 
 There are some example pictures in ./data, so you could just run the command to test whether it  work.

## Test
 CUDA_VISIBLE_DEVICES=0 python3 test.py --datadir ./data/test  --model segnet --model-dir ./models/ --save-dir ./results/

## Detail
1. More train and test options see ./options
2. datadir include image.txt and label.txt, and the default datasets is for [pascalvoc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
3. If you want train your own data. Remember to modify your data labels in ./utils/cityscapes/helpers/labels,and NUM_CLASSES in options.
4. You can change the way the model load data in ./dataloader/ to fit your format of dataset. 
5. test.py would calculate the mIoU and save the result of segmented picture in --savedir.
