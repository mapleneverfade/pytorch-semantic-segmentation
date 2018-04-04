import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png','.JPG','.PNG']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, '{}{}'.format(basename,extension))

def image_path_city(root, name):
    return os.path.join(root, '{}'.format(name))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])



class NeoData(Dataset):
    def __init__(self,transform=None, imagedata=None,labeldata=None):
        #  make sure label match with image
        self.image = imagedata
        self.label = labeldata
        
        self.transform = transform 

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]
        
        with open(filename, 'rb') as f: 
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.image)
    

