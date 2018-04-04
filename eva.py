import time
import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from utils import evalIoU

def eval(args, model, loader_val, criterion, epoch):
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        #New confusion matrix 
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
            if (args.iouVal):

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
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print('VAL loss: {} (epoch: {}, step: {})'.format(average,epoch,step), 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        # Calculate val IOU scores on class level from matrix
        iouVal = 0
        if (args.iouVal):
            #start_time_iou = time.time()
            classScoreList = {}
            for label in evalIoU.args.evalLabels:
                labelName = evalIoU.trainId2label[label].name
                classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)

            iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
            iouVal = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
            print ("EPOCH IoU on VAL set: ", iouAvgStr)
        return average_epoch_loss_val, iouVal
