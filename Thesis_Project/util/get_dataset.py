import os
import tensorflow as tf
import functools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def getFileName(file_dir,AddDirectory=False):
    '''
    Get the list of all the files' name in the given directory
    :param file_dir: File directory
    :type file_dir: str
    :param AddDirectory: Whether the return list contains the directory
    :type AddDirectory: bool
    :return the files' name list
    '''
    for _,_,files in os.walk(file_dir):

        fileNameList=[]
        if AddDirectory == True:
            for file in files:
                filename=os.path.join(file_dir,file)
                fileNameList.append(filename)
        else:
            for file in files:
                fileNameList.append(file)
    return fileNameList

def _parse_function(img_name, mask_name,img_shape,Resize=None):
    """
    Read the image and its mask based their path
    :param img_name: image path
    :param mask_name: mask image path
    :param img_shape: image shape
    :param Resize: If assigned, then resize images into corresponding size
    :return: image and its mask tensors
    """
    image_string = tf.read_file(img_name)
    mask_string = tf.read_file(mask_name)

    image_decoded = tf.image.decode_image(image_string,channels=3)
    mask_decoded= tf.image.decode_image(mask_string,channels=1)
    

    image_decoded.set_shape(img_shape)
    mask_decoded.set_shape([img_shape[0],img_shape[1],1])
    print("before resize",image_decoded[0,0,0].dtype)
    if Resize is not None:
        image_decoded=tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        mask_decoded=tf.image.convert_image_dtype(mask_decoded, dtype=tf.float32)
        print("first convert",image_decoded[0,0,0].dtype)
        image_decoded=tf.image.resize_images(image_decoded,size=Resize,method=1)
        mask_decoded=tf.image.resize_images(mask_decoded,size=Resize,method=1)
        print("after resize",image_decoded[0,0,0].dtype)
        image_decoded=tf.image.convert_image_dtype(image_decoded, dtype=tf.uint8)
        mask_decoded=tf.image.convert_image_dtype(mask_decoded, dtype=tf.uint8)
        print("second convert",image_decoded[0,0,0].dtype,image_decoded.get_shape())

    return image_decoded,mask_decoded

def getDataset(img_name,mask_name,img_shape,resize=None):
    """
    Get the dataset based on image directory
    :param img_name: directory of images
    :type img_name: list
    :param mask_name: directory of mask images
    :type mask_name: list
    :param img_shape: the shape of image
    :type img_shape: list
    :return: A tf.dataset containing images and their masks
    """

    tf_img_name=tf.constant(img_name)
    tf_mask_name=tf.constant(mask_name)
    dataset=tf.data.Dataset.from_tensor_slices((tf_img_name,tf_mask_name))
    dataset=dataset.map(lambda x,y:_parse_function(x,y,img_shape,Resize=resize))
    return dataset
