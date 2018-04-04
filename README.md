# Pytorch-Semantic-Segmentation
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
- pytorch 0.2.0
- torchvision 0.2.0
- python 3.5.2  

## Download
Recommand you use virtualenv.
> virtualenv -p python3 YourVirtualEnv --no-site-packages  

> git clone https://github.com/mapleneverfade/pytorch-semantic-segmentation.git  

## Train 
 If gpu is available:  
 
 `CUDA_VISIBLE_DEVICES=0 python3 train.py --datadir ./data/train --savedir ./save_models/ --model segnet`  
 else:  
 
 `python3 train.py --cuda False --datadir ./data/train --savedir ./save_models/ --model segnet`  
 
 There are some example pictures in ./data, so you could just run the command to test whether it  work.

## Test
 `CUDA_VISIBLE_DEVICES=0 python3 test.py --datadir ./data/test  --model segnet --model-dir ./save_models/segnet_50.pth --save-dir ./results/`  
 "--model-dir" is the path to your trained model

## Detail
1. More train and test options see ./options
2. datadir include image.txt and label.txt, and the default datasets is for [pascalvoc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
3. If you want train your own data. Remember to modify your data labels in ./utils/cityscapes/helpers/labels,and NUM_CLASSES in options.
4. You can change the way the model load data in ./dataloader/ to fit your format of dataset. 
5. test.py would calculate the mIoU and save the result of segmented picture in --savedir.

## ToDo
1. More Network
2. Clean up the code.
