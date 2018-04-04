from PIL import Image,ImageOps
import os
import numpy as np


imagefile = []
with open('./label.txt_all','r') as f:
	for line in f:
		imagefile.append(line.strip().replace('\n',''))



s = 0
light = 0
car = 0
count=0
background = 0
print(len(imagefile))
for i in imagefile:
		img = np.array(Image.open(i).convert('P'))
		count+=1
		background += np.sum(img==0)

		car += np.sum(img==1)
		light += np.sum(img==2)

		s += (img.shape[0]*img.shape[1])
		print(count)


print(background/s,light/s,car/s)
weight_back = 1/np.log(1+background/s)
weight_car = 1/np.log(1+car/s)
weight_light = 1/np.log(1+light/s)
print(weight_back,weight_car,weight_light)        #1,1345.54,4779.0