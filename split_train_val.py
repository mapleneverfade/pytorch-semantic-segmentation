import os
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


def split_train_val(args):
    imagepath = args.imagepath
    labelpath = args.labelpath
    assert os.path.exists(imagepath), "{} is not exists!".format(imagepath)
    assert os.path.exists(labelpath), "{} is not exists!".format(labelpath)
    
    image = []
    label = []
    with open(os.path.join(args.savedir,'image.txt'),'r') as f:
        for line in f:
            image.append(line.strip())
    with open(os.path.join(args.savedir,'label.txt'),'r') as f:
        for line in f:
            label.append(line.strip())
            
    image_train, image_val, label_train, label_val = train_test_split(image,label,random_state=args.random_state,train_size=args.train_size,test_size=args.val_size)
    
    #store train data in ./data/train/image.txt
    with open(os.path.join(args.savedir,'train/image.txt'),'w') as f:
        for image in image_train:
            f.write(image+'\n')
    with open(os.path.join(args.savedir,'train/label.txt'),'w') as f:
        for label in label_train:
            f.write(label+'\n')
            
    #store val data in ./data/val/image.txt
    with open(os.path.join(args.savedir,'val/image.txt'),'w') as f:
        for image in image_val:
            f.write(image+'\n')
    with open(os.path.join(args.savedir,'val/label.txt'),'w') as f:
        for label in label_val:
            f.write(label+'\n')
    print('Done!')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--savedir', default='./data/')
    parser.add_argument('--imagepath', default='./data/') 
    parser.add_argument('--labelpath', default='./data/')
    parser.add_argument('--random-state',default=10000)
    parser.add_argument('--train-size',default=0.7)
    parser.add_argument('--val-size',default=0.3)
    
    split_train_val(parser.parse_args())
