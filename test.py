import os
import time
import torch
from options.test_options import TestOptions
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor,Normalize,ToPILImage
from PIL import Image
from utils import evalIoU
from utils.label2Img import label2rgb
from dataloader.transform import Transform_test
from dataloader.dataset import NeoData_test
from networks import get_model
from eval import *

def main(args):
    ######## load the test data ##########    
    despath = args.savedir
    
    if not os.path.exists(despath):
        os.mkdir(despath)

    imagedir = os.path.join(args.datadir,'image.txt')     
    labeldir = os.path.join(args.datadir,'label.txt')            
                                         
    image_test = []
    label_test = [] 
    with open(imagedir,'r') as f:
        for line in f:
            image_test.append(line.strip().replace('\n',''))
    with open(labeldir,'r') as f:
        for line in f:
            label_test.append(line.strip().replace('\n',''))

    print("length of testimage and its label: {}".format(len(image_test)))

    transform = Transform_test(args.size)
    dataset_test = NeoData_test(transform, image_test, label_test)
    loader = DataLoader(dataset_test, num_workers=4, batch_size=1,shuffle=False) #test data loader

    #eval the result of IoU
    time_train = []
    confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
    perImageStats = {}
    nbPixels = 0
    usedLr = 0
    
    model = get_model(args)
    if args.cuda:
        model = Net(args.num_classes).cuda()
    model.load_state_dict(torch.load(args.model_dir))
    model.eval()
    count = 0
    for step, colign in enumerate(zip(loader,image_test)):
 
      img = colign[0][2].squeeze(0).numpy()       #image-numpy,original image   
      images = colign[0][0]                       #image-tensor
      label = colign[0][1]                        #label-tensor

      if args.cuda:
        images = images.cuda()
      inputs = Variable(images,volatile=True)

      outputs = model(inputs)
      out = outputs[0].cpu().max(0)[1].data.squeeze(0).byte().numpy() #index of max-channel 
    
      add_to_confMatrix(outputs, label, confMatrix, perImageStats, nbPixels)

      label2img = label2rgb(out,img,n_labels = args.num_classes)   #show segmented result
      Image.fromarray(label2img).save(despath + '_label2img_' + colign[1].strip().split('/')[-1])
      count += 1
      print("This is the {}th of image!".format(count))
        
    iouAvgStr, iouTest, classScoreList = cal_iou(evalIoU, confMatrix)
    print("IoU on TEST set : ",iouAvgStr)
    #print("IoU on TEST set of each class - car:{}  light:{} ".format(classScoreList['car'],classScoreList['light']))

if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)


