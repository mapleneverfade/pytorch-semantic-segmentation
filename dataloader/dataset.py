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
    def __init__(self, imagepath=None, labelpath=None, transform=None):
        #  make sure label match with image 
        self.transform = transform 
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)                                  
        self.image = []
        self.label= [] 
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip().replace('\n',''))
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip().replace('\n',''))

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
    
class NeoData_test(Dataset):
    def __init__(self, imagepath=None, labelpath=None, transform=None):
        self.transform = transform 
        
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        
        self.image = []
        self.label= [] 
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())#.replace('\n',''))
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())#.replace('\n',''))
        print("Length of test data is {}".format(len(self.image)))
    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]
        
        with open(filename, 'rb') as f: # advance
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.transform is not None:
            image_tensor, label_tensor, img = self.transform(image, label)

        return (image_tensor, label_tensor, np.array(img))  #return original image, in order to show segmented area in origin

    def __len__(self):
        return len(self.image)

