#-*- coding:utf-8 -*-
import argparse
import os

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--model', default="erfnet", help='model to train,options:fcn8,segnet...')
        self.parser.add_argument('--state')    
        self.parser.add_argument('--num-classes', type=int, default=21)
        self.parser.add_argument('--datadir', default="./data/train/",help='path where image.txt and label.txt lies')
        self.parser.add_argument('--savedir', type=str, default='./save_models/',help='savedir for models')
        self.parser.add_argument('--lr', type=float, default=5e-4)
        self.parser.add_argument('--num-epochs', type=int, default=150)
        self.parser.add_argument('--num-workers', type=int, default=4)
        self.parser.add_argument('--batch-size', type=int, default=1)
        self.parser.add_argument('--epoch-save', type=int, default=5)    #You can use this value to save model every X epochs
        self.parser.add_argument('--iouTrain', action='store_true', default=True) #recommended: False (takes a lot to train otherwise)
        self.parser.add_argument('--iouVal', action='store_true', default=True) #calculating IoU   
        self.parser.add_argument('--steps-loss', type=int, default=5)
        self.parser.add_argument('--pretrained',type=str, default='./pre_trained/~~~.pth') 
        self.parser.add_argument('--resume', action='store_true', default= False)   
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
