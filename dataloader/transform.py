import numpy as np
import torch
from .functional import RandomCrop, CenterCrop,RandomFlip,RandomRotate
from PIL import Image
import random
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Normalize
def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    for i in np.arange(n):
        r, g, b = np.zeros(3)
        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))
        cmap[i,:] = np.array([r, g, b])
    return cmap

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel
    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:
    def __call__(self, image):  
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)  #np.array change the size of image

class Colorize:
    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        return color_image

class MyTransform(object):
    '''
        1. self-define transform rules, including resize, crop, flip. (crop and flip only for training set)
                2.   training set augmentation with RandomCrop and RandomFlip.
                3.   validation set using CenterCrop
    '''
    def __init__(self,reshape_size=None, crop_size = None , augment=True): 
        self.reshape_size = reshape_size
        self.crop_size = crop_size
        self.augment = augment
        self.flip = RandomFlip()
        self.rotate = RandomRotate(32)
       
        self.count = 0
    def __call__(self, input, target):
        # do something to both images and labels
        if self.reshape_size is not None:
            input = input.resize(self.reshape_size,Image.BILINEAR)
            target = target.resize(self.reshape_size,Image.NEAREST)
 
        if self.augment :
            input, target = RandomCrop(self.crop_size)(input,target) # RandomCrop for  image and label in the same area
            input, target = self.flip(input,target)               # RandomFlip for both croped image and label
            input, target = self.rotate(input,target)
        else:
            input, target =  CenterCrop(self.crop_size)(input, target) # CenterCrop for the validation data
            
        input = ToTensor()(input)  
        Normalize([.485, .456, .406], [.229, .224, .225])(input) #normalize with the params of imagenet
          
        target = torch.from_numpy(np.array(target)).long().unsqueeze(0)

        return input, target
    
class Transform_test(object):
    '''
        Transform for test data.Reshape size is difined in ./options/test_options.py
    '''
    def __init__(self,size):
        self.size = size   
    def __call__(self, input, target):
        # do something to both images 
        input =  input.resize(self.size, Image.BILINEAR)
        target = target.resize(self.size,Image.NEAREST)

        target = torch.from_numpy(np.array(target)).long().unsqueeze(0)
        input_tensor = ToTensor()(input)  
        Normalize([.485, .456, .406], [.229, .224, .225])(input_tensor)
        return input_tensor, target, input
        
def img2label(img,label,count):
    count+=1
    img = np.array(img)
    label = np.array(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j]==0:
                img[i,j,:]=0
    image = ToPILImage()(img)
    image.save('./results/imglabel_'+str(count)+'.jpg')
