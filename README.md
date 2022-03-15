# wrench-detection-using-keras-and-tensorflow
In this project I have created wrench detection model using keras. I created my own dataset by cropping (openCV), copying and pasting wrenches located in keys folder,
and pasting it into a background images located in background folder in a random way (random angle and location of the given wrench). Bounding box coordinates are 
obtained by knowing the location of pasted wrench, so in the end xml annotation files are not needed. This is quite unique as most of object detection models use
xml's or programs that create xml annotation files. I use VGG16 to train the model. In the end, model is tested on test set - it's quasi-efective. It reaches accuracy 
of 70% on a train set which consists of 200 images. There are two ways to improve this project. Creating bigger dataset and longer training time. 
