#-*- coding:utf-8 -*-
import argparse
import os

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--model', default="erfnet", help='model to train,options:fcn8,segnet...')  
        self.parser.add_argument('--model-dir', default="./models/", help='path to stored-model')   
        self.parser.add_argument('--num-classes', type=int, default=21)
        self.parser.add_argument('--datadir', default="./data/test/",help='path where image.txt and label.txt lies')
        
        self.parser.add_argument('--stored',default=True, help='whether or not store the result')
        self.parser.add_argument('--savedir', type=str, default='./results/',help='options. visualize the result of segmented picture, not just show IoU')
        
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
