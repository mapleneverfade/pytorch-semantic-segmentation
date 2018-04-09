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
- cython

## Download
Recommand you use virtualenv.
> virtualenv -p python3 YourVirtualEnv --no-site-packages  

> git clone https://github.com/mapleneverfade/pytorch-semantic-segmentation.git  
## Install CSUPPORT (Options)
To speed up calculating IoU:
> cd ./utils/cityscape/    

> python setup.py install
## Train 
 If gpu is available:  
 
 `CUDA_VISIBLE_DEVICES=0 python3 train.py --datadir ./data/ --savedir ./save_models/ --model segnet`  
 else:  
 
 `python3 train.py --cuda False --datadir ./data/ --savedir ./save_models/ --model segnet`  
 
 There are some example pictures in ./data, so you could just run the command to test whether it  work.
### More Training Options
    --model        model to use ['segnet fcn8 fcn16 fcn32 erfnet pspnet unet'] [default=segnet]·
    --datadir      where you store [train/val].In my case, './data' should have subfolders './data/train/' and './data/val/' where store image.txt' and 'label.txt'. [default='./data/']
    --savedir      path to savedir [default='./save_models/']
    --lr           learning rate [default=5e-4]·
    --num-epochs   epochs [default=150]·
    --num-classes  number of labels, pascalvoc is 21, cityscape is 20. change it when you training your own dataset.[default=21] 

## Test
 `CUDA_VISIBLE_DEVICES=0 python3 test.py --datadir ./data/test  --model segnet --model-dir ./save_models/segnet_50.pth --save-dir ./results/`  
 "--model-dir" is the path to your trained model
### More Testing Options
    --model        [default=segnet]
    --model-dir    path to your trained model. For example './save_models/segnet/segnet_epoch_5.pth'
    --datadir      [default='./data/test/']
    --num-classes  number of labels, pascalvoc is 21, cityscape is 20. change it when you training your own dataset.[default=21]
    --size         reshape size [default=(672,480)]
    --savedir      [default='./results/']
## Options
split_train_val.py mainly to split original [image.txt,label.txt] into './train/[image.txt,label.txt]' and './val/[image.txt,label.txt]'  

Options for split:  

    --savedir        [default='./data/']
    --imagepath      path to your own image.txt
    --labelpath      path to your own label.txt
    --random-state   random seed [default=10000]
    --train-size     ratio of train set [default=0.7]
    --val-size       ratio of val set   [default=0.3]
For example, if your original folder like this :  

    ./data
          image.txt
          label.txt
After run 'python3 split_train_val.py --savedir ./data --imagepath ./data/image.txt --labelpath ./data/label.txt', you will see this:  

    ./data
          ./train
                 image.txt
                 label.txt
          ./val   
                 image.txt
                 label.txt
                                                      
## Detail
1. More train and test options see ./options
2. datadir include image.txt and label.txt, and the default datasets is for [pascalvoc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
3. If you want train your own data. Remember to modify your data labels in ./utils/cityscapes/helpers/labels,and NUM_CLASSES in options.
4. You can change the way the model load data in ./dataloader/ to fit your format of dataset. 
5. test.py would calculate the mIoU and save the result of segmented picture in --savedir.

## ToDo
1. More Network
2. Clean up the code.
