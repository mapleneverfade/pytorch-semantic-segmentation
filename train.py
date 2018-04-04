import os 
import time
import math
import torch
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
from sklearn.model_selection import train_test_split

NUM_CHANNELS = 3

def train(args, model):
    NUM_CLASSES = args.num_classes #pascal=21, cityscapes=20

    weight = torch.ones(NUM_CLASSES)
    
    #add the weight of each class (1/ln(c+Pclass))
    #calculate the weights of each class

    #weight[0]=1.45
    ##weight[1]=54.38
    #weight[2] = 428.723
    imagedir = os.path.join(args.datadir,'image.txt')     
    labeldir = os.path.join(args.datadir,'label.txt')            
                                         
    image_train = []
    label_train= [] 

    with open(imagedir,'r') as f:
        for line in f:
            image_train.append(line.strip().replace('\n',''))
    with open(labeldir,'r') as f:
        for line in f:
            label_train.append(line.strip().replace('\n',''))

    # options,  random split dataset into train,val,test, and record test data into '.txt'
    image_train, image_val, label_train, label_val = train_test_split(image_train,label_train,random_state=10254,train_size=0.5,test_size=0.5)
    
    print('training set is {} '.format(len(image_train)))
    print('val set is {} '.format(len(image_val)))

    # my picture size is too big, so i resize then first before crop (732,512)
    train_transform = MyTransform(reshape_size=(500,350),crop_size=(448,320), augment=True)  # data transform for training set with data augmentation, including resize, crop, flip and so on
    val_transform = MyTransform(reshape_size=(500,350),crop_size=(448,320), augment=False)   #data transform for validation set without data augmentation

    dataset_train = NeoData(train_transform,image_train,label_train) # self-define dataset , you can change the way it work .
    dataset_val = NeoData(val_transform,image_val, label_val)

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight).cuda() 
    else:
        criterion = CrossEntropyLoss2d(weight)

    savedir = args.savedir

    #save log
    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))
    
    optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
    #optimizer = nn.DataParallel(optimizer) #multi-gpu

    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch            
        
    start_epoch = 1    
    for epoch in range(start_epoch, args.num_epochs+1):

        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)   

        epoch_loss = []
        time_train = []
        
        #calculate IoU
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      
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
            
            #print('size is {} target size is {}'.format(inputs.size(), targets.size()))

            outputs = model(inputs) 
            
            #print('outputs size is {}'.format(outputs.size()))
            loss = criterion(outputs, targets[:, 0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.data[0])
            time_train.append(time.time() - start_time)

            #Add outputs to confusion matrix    #CODE USING evalIoU.py remade from cityscapes/scripts/evaluation/evalPixelLevelSemanticLabeling.py
            if (doIouTrain):
                #compatibility with criterion dataparallel
                if isinstance(outputs, list):   #merge gpu tensors
                    outputs_cpu = outputs[0].cpu()
                    for i in range(1,len(outputs)):
                        outputs_cpu = torch.cat((outputs_cpu, outputs[i].cpu()), 0)
                else:
                    outputs_cpu = outputs.cpu()

                for i in range(0, outputs_cpu.size(0)):   #args.batch_size
                    prediction = ToPILImage()(outputs_cpu[i].max(0)[1].data.unsqueeze(0).byte())
                    groundtruth = ToPILImage()(labels[i].cpu().byte())
                    nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth, confMatrix, perImageStats, evalIoU.args)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {} (epoch: {}, step: {})'.format(average,epoch,step), 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        #evalIoU.printConfMatrix(confMatrix, evalIoU.args)
        
        iouTrain = 0
        if (doIouTrain):
            # Calculate IOU scores on class level from matrix
            classScoreList = {}
            for label in evalIoU.args.evalLabels:
                labelName = evalIoU.trainId2label[label].name
                classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)
            iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
            
            iouTrain = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
            print ("EPOCH IoU on TRAIN set: ", iouAvgStr)
            #print("train class IoU airline:{}  dog:{}".format(classScoreList['aeroplane'],classScoreList['dog']))

            #print("train class IoU car:{}  light:{}".format(classScoreList['car'],classScoreList['light']))

            
        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        #New confusion matrix data
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        
        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            inputs = Variable(images, volatile=True)    
            targets = Variable(labels, volatile=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets[:, 0])
            
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)

            #Add outputs to confusion matrix
            if (doIouVal):
                #compatibility with criterion dataparallel
                if isinstance(outputs, list):   #merge gpu tensors
                    outputs_cpu = outputs[0].cpu()
                    for i in range(1,len(outputs)):
                        outputs_cpu = torch.cat((outputs_cpu, outputs[i].cpu()), 0)
                    #print(outputs_cpu.size())
                else:
                    outputs_cpu = outputs.cpu()

                #start_time_iou = time.time()
                for i in range(0, outputs_cpu.size(0)):   #args.batch_size
                    prediction = ToPILImage()(outputs_cpu[i].max(0)[1].data.unsqueeze(0).byte())
                    groundtruth = ToPILImage()(labels[i].cpu().byte())
                    nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth, confMatrix, perImageStats, evalIoU.args)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print('VAL loss: {} (epoch: {}, step: {})'.format(average,epoch,step), 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        # Calculate IOU scores on class level from matrix
        iouVal = 0
        if (doIouVal):
            #start_time_iou = time.time()
            classScoreList = {}
            for label in evalIoU.args.evalLabels:
                labelName = evalIoU.trainId2label[label].name
                classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)

            iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
            iouVal = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
            print ("EPOCH IoU on VAL set: ", iouAvgStr)
           # print("val class IoU car:{}  light:{}".format(classScoreList['car'],classScoreList['light']))

        #save model every X epoch
        print(epoch)
        if  epoch % args.epoch_save==0:
            torch.save(model.state_dict(), '{}_{}.pth'.format(os.path.join(args.savedir,args.model),str(epoch)))

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   
    
def main(args):
    savedir = '{}'.format(args.savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))
        
    model = get_model(args)
    if args.cuda:
        # model = torch.nn.DataParallel(model).cuda()  #multi-gpu
        model = model.cuda() 
        
    print("========== TRAINING ===========")
    
    model = train(args,model)

    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':

    parser = TrainOptions().parse()
    main(parser)
