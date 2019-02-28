import os

from util.get_img_masks import *
from util.get_dataset import *
from util.dataset2tfrecords import *

#Get image mask
#Here the label is in COCO format
#ref:http://cocodataset.org/#format-data
#The goal of this step is to obtain the image label, pixel-wise label images with same size
#If you already have them, skip this step.

data_directory=r"data_directory"
TRAIN_IMAGES_DIRECTORY = "TRAIN_IMAGES_DIRECTORY"
TRAIN_MASKS_DIRECTORY = "TRAIN_MASKS_DIRECTORY"
TRAIN_ANNOTATIONS_SMALL_PATH = "TRAIN_ANNOTATIONS_SMALL_PATH"

getImageMasks(TRAIN_IMAGES_DIRECTORY,TRAIN_MASKS_DIRECTORY,TRAIN_ANNOTATIONS_SMALL_PATH)


#Get dataset from name list

#Obtain the files' path as lists 
trainMaskImgName=getFileName(TRAIN_MASKS_DIRECTORY,AddDirectory=False)
trainImgName=trainMaskImgName.copy()
trainImgName=list(map(lambda x:os.path.join(TRAIN_IMAGES_DIRECTORY,x),trainImgName))
trainMaskImgName=getFileName(TRAIN_MASKS_DIRECTORY,AddDirectory=True)

#Then obtain tf.data.Dataset
img_shape = (256, 256, 3)
data_num=100
offset=0
dataset=getDataset(trainImgName[offset:offset+data_num],trainMaskImgName[offset:offset+data_num],[300,300,3],resize=[256,256])

#Get TFRecordfile from tf.data.Dataset
output_dir="output_dir"
output_name="output_name"
dataset2tfrecords(dataset, output_name, output_dir)
