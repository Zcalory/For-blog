import os

from util.get_img_masks import *
from util.get_dataset import *
from util.data2tfrecords import *

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

#Obtain the files' name as a list
name_list = getFileName(TRAIN_MASKS_DIRECTORY,AddDirectory=False)
#print(len(name_list))

resize = (256,256)
data_num=0
offset=0
output_dir="output_dir"
output_name="output_name"
data2tfrecords(name_list[offset:offset+data_num],TRAIN_IMAGES_DIRECTORY,TRAIN_MASKS_DIRECTORY, output_name,output_dir,Resize = resize)

