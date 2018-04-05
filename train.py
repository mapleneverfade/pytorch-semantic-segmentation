import os 
import time
import math
import torch
from eval import *
import torch.nn as nn
from utils import evalIoU
from networks import get_model
from torch.autograd import Variable
from dataloader.dataset import NeoData
from torch.utils.data import DataLoader
from dataloader.transform import MyTransform
from torchvision.transforms import ToPILImage
from options.train_options import TrainOptions
from torch.optim import SGD, Adam, lr_scheduler
from criterion.criterion import CrossEntropyLoss2d
NUM_CHANNELS = 3

def get_loader(args):
    #add the weight of each class (1/ln(c+Pclass))
    #calculate the weights of each class

    #weight[0]=1.45
    ##weight[1]=54.38
    #weight[2] = 428.723
    imagepath_train = os.path.join(args.datadir, 'train/image.txt')
    labelpath_train = os.path.join(args.datadir, 'train/label.txt')
    imagepath_val = os.path.join(args.datadir, 'val/image.txt')
    labelpath_val = os.path.join(args.datadir, 'val/label.txt')
    
    train_transform = MyTransform(reshape_size=(500,350),crop_size=(448,320), augment=True)  # data transform for training set with data augmentation, including resize, crop, flip and so on
    val_transform = MyTransform(reshape_size=(500,350),crop_size=(448,320), augment=False)   #data transform for validation set without data augmentation
    
    dataset_train = NeoData(imagepath_train, labelpath_train, train_transform) #DataSet
    dataset_val = NeoData(imagepath_val, labelpath_val, val_transform)
    
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    return loader, loader_val

def train(args, model):
    NUM_CLASSES = args.num_classes #pascal=21, cityscapes=20
    savedir = args.savedir
    weight = torch.ones(NUM_CLASSES)

    loader, loader_val = get_loader(args)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight).cuda() 
    else:
        criterion = CrossEntropyLoss2d(weight)
        
    #save log
    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")
    
    optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch            
    start_epoch = 1   
    
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        
        scheduler.step(epoch)   
        epoch_loss = []
        time_train = []
        
        #confmatrix for calculating IoU   
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        usedLr = 0
        #for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        count = 1
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            
            outputs = model(inputs) 
            loss = criterion(outputs, targets[:, 0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.data[0])
            time_train.append(time.time() - start_time)

            #Add outputs to confusion matrix    #CODE USING evalIoU.py remade from cityscapes/scripts/evaluation/evalPixelLevelSemanticLabeling.py
            if (args.iouTrain):
                add_to_confMatrix(outputs, labels, confMatrix, perImageStats, nbPixels)
              
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {} (epoch: {}, step: {})'.format(average,epoch,step), 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouAvgStr, iouTrain, classScoreList = cal_iou(evalIoU, confMatrix)
        print ("EPOCH IoU on TRAIN set: ", iouAvgStr)
                       
        # calculate eval-loss and eval-IoU
        average_epoch_loss_val, iouVal = eval(args, model, loader_val, criterion, epoch)     
        
        #save model every X epoch
        if  epoch % args.epoch_save==0:
            torch.save(model.state_dict(), '{}_{}.pth'.format(os.path.join(args.savedir,args.model),str(epoch)))

        #save log
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   
    
def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = '{}'.format(args.savedir)
    modeltxtpath = os.path.join(savedir,'model.txt') 

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile: #record options
        myfile.write(str(args))
        
    model = get_model(args)     #load model
    
    with open(modeltxtpath, "w") as myfile:  #record model 
        myfile.write(str(model))
        
    if args.cuda:
        model = model.cuda() 
    print("========== TRAINING ===========")
    model = train(args,model)
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':

    parser = TrainOptions().parse()
    main(parser)
