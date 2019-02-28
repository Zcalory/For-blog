import tensorflow as tf
import os
from PIL import Image

def data2tfrecords(name_list,img_dir,mask_dir,tfrecords_filename,tfrecords_file_dir,Resize = None):
    """
    Tranform image data into TFRecord file
    :param name_list: image files's name
    :type name_list: list
    :param img_dir: image directory
    :type img_dir: str
    :param mask_dir: mask image directory
    :type mask_dir:str
    :param tfrecords_filename: name of tfrecords file
    :type tfrecords_filename
    :param tfrecords_file_dir: save directory of tfrecords file
    :type tfrecords_file_dir: str
    :return: None
    """
    #It is suggested that each TFRecord file just contains 1000 images
    maxnum=1000/2
    tfrecordfilenum=0
    imgnum=0
    cur_tffilename=(tfrecords_filename+".tfrecords-%.3d" % tfrecordfilenum)
    cur_tffilename=os.path.join(tfrecords_file_dir,cur_tffilename)
    writer = tf.python_io.TFRecordWriter(cur_tffilename)
    if not os.path.exists(tfrecords_file_dir):
        os.makedirs(tfrecords_file_dir)
        
        
    print("Start transforming...")
    for item in name_list:
        #print(item)
        try:
            #print(os.path.join(img_dir,item))
            img  = Image.open(os.path.join(img_dir,item))
            mask = Image.open(os.path.join(mask_dir,item))
            
            #resize image and mask
            if Resize is not None:
                img = img.resize(Resize,resample=Image.NEAREST)
                mask = mask.resize(Resize, resample=Image.NEAREST)
            
            imgnum+=1
            if imgnum>maxnum:
                imgnum=1
                tfrecordfilenum+=1
                cur_tffilename=(tfrecords_filename+".tfrecords-%.3d" % tfrecordfilenum)
                cur_tffilename=os.path.join(tfrecords_file_dir,cur_tffilename)
                writer = tf.python_io.TFRecordWriter(cur_tffilename)

            img_data = img.tobytes() 
            mask_data = mask.tobytes() 
            #print(type(img_data))
        
            example=tf.train.Example(features=tf.train.Features(
                    feature={
                    'shape': tf.train.Feature(int64_list = tf.train.Int64List(value=[img.size[0],img.size[1],len(img.getbands())])),
                    'img_data':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_data])),
                    'mask_data':tf.train.Feature(bytes_list = tf.train.BytesList(value=[mask_data]))
                    }))
            writer.write(example.SerializeToString())
        except Exception as e:
            print(str(e))
            break
    print("Finish")
    writer.close()
    return 0
