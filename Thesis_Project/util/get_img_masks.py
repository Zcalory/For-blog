from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os

def getSegmentationsList(annotations):
    segList=[]
    for ann in annotations:
        segList.append(ann['segmentation'][0])
    return segList

def getImageMasks(imgDirectory,maskDirectory,cocoFilePath):
    '''
    Obtain the mask images based on their annotation(in COCO format) and save them in the given directory.
    :param imgDirectory: The directory of images
    :type imgDirectory: string
    :param maskDirectory: The directory to save the mask images
    :type maskDirectory: string
    :param cocoFilePath: The directory of the COCO file
    :type cocoFilePath: string
    '''
    coco= COCO(cocoFilePath)
    cat_id=coco.loadCats(coco.getCatIds())
    image_ids=coco.getImgIds(catIds=coco.getCatIds())
    imgs=coco.loadImgs(image_ids)
    if not os.path.exists(maskDirectory):
        os.makedirs(maskDirectory)
    for img in imgs:
        image_path = os.path.join(imgDirectory, img["file_name"])
        I = io.imread(image_path)
        #print(I.shape)
        #%matplotlib inline
        #plt.imshow(I)
        #plt.show()
        
        ann_ids=coco.getAnnIds(imgIds=img['id'])
        ann=coco.loadAnns(ann_ids)
        
        seglist=getSegmentationsList(ann)
        
        #%matplotlib inline
        #plt.imshow(I)
        #coco.showAnns(ann)
        #plt.show()
        rle = cocomask.frPyObjects(seglist, img['height'], img['width'])
        merge_rle=cocomask.merge(rle,intersect=False)
        
        m=cocomask.decode(merge_rle)
        m=m.reshape((img['height'], img['width']))
        #background is 0, mask is 1
        #multiply 255 to make the pixel value of mask be 255
        #m[m<1]=0
        #print(type(m),m.dtype)
        #break
        #mask=m.copy()
        mask=np.ones((img['height'], img['width']),dtype=np.float32)
        mask[m==0]=0
        #m=m*255
        #mask=np.zeros((img['height'], img['width'],2))
        #mask[:,:,0]=m
        #mask[:,:,1]=1-m
        saveName=os.path.join(maskDirectory, img['file_name'])
        #print(mask.shape)
        #print(m[0:30,0:30])
        io.imsave(saveName,mask)
    print("End")
