# Something about dataset.
1. NeoData in dataset.py, mainly to load the data(both image and label), transform data which defined by transform.py.  

2. You can change the way it work. I stored all the image path in '../data/train/image.txt',label path in '../data/train/label.txt'  

3. Transform mainly include **Resize**, **Crop** and **Flip** the data. You would decide whether it is necessary to resize or not.  

4. For my data it is too large, so i resize them before crop. For training data, using **RandomCrop**, for val data, using **CenterCrop**.
