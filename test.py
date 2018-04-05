import os
import time
import torch
from options.test_options import TestOptions
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.label2Img import label2rgb
from dataloader.transform import Transform_test
from dataloader.dataset import NeoData_test
from networks import get_model
from eval import *

def main(args):
    despath = args.savedir
    if not os.path.exists(despath):
        os.mkdir(despath)

    imagedir = os.path.join(args.datadir,'image.txt')     
    labeldir = os.path.join(args.datadir,'label.txt')            
                                         
    transform = Transform_test(args.size)
    dataset_test = NeoData_test(imagedir, labeldir, transform)
    loader = DataLoader(dataset_test, num_workers=4, batch_size=1,shuffle=False) #test data loader

    #eval the result of IoU
    confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
    perImageStats = {}
    nbPixels = 0
    usedLr = 0
    
    model = get_model(args)
    if args.cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(args.model_dir))
    model.eval()
    count = 0
    for step, colign in enumerate(loader):
 
      img = colign[2].squeeze(0).numpy()       #image-numpy,original image   
      images = colign[0]                       #image-tensor
      label = colign[1]                        #label-tensor

      if args.cuda:
        images = images.cuda()
      inputs = Variable(images,volatile=True)

      outputs = model(inputs)
      out = outputs[0].cpu().max(0)[1].data.squeeze(0).byte().numpy() #index of max-channel 
      
      add_to_confMatrix(outputs, label, confMatrix, perImageStats, nbPixels)  #add result to confusion matrix

      label2img = label2rgb(out,img,n_labels = args.num_classes)   #merge segmented result with original picture 
      Image.fromarray(label2img).save(despath + 'label2img_' +str(count)+'.jpg' )
      count += 1
      print("This is the {}th of image!".format(count))
        
    iouAvgStr, iouTest, classScoreList = cal_iou(evalIoU, confMatrix)  #calculate mIoU, classScoreList include IoU for each class
    print("IoU on TEST set : ",iouAvgStr)
    #print("IoU on TEST set of each class - car:{}  light:{} ".format(classScoreList['car'],classScoreList['light']))

if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)


