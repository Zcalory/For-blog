import tensorflow as tf
import os

def dataset2tfrecords(dataset,tfrecords_filename,tfrecords_file_dir):

	#It is suggested that each TFRecord file just contains 1000 images
    maxnum=1000/2
    tfrecordfilenum=0
    imgnum=0
    cur_tffilename=(tfrecords_filename+".tfrecords-%.3d" % tfrecordfilenum)
    cur_tffilename=os.path.join(tfrecords_file_dir,cur_tffilename)
    writer = tf.python_io.TFRecordWriter(cur_tffilename)
    if not os.path.exists(tfrecords_file_dir):
        os.makedirs(tfrecords_file_dir)
        
    dataset_iter=dataset.make_one_shot_iterator()
    next_ele=dataset_iter.get_next()

    with tf.Session() as sess:
        print("start transforming...")
        while True:
            try:
                img,mask=sess.run(next_ele)
                imgnum+=1
                if imgnum>maxnum:
                    imgnum=1
                    tfrecordfilenum+=1
                    cur_tffilename=(tfrecords_filename+".tfrecords-%.3d" % tfrecordfilenum)
                    cur_tffilename=os.path.join(tfrecords_file_dir,cur_tffilename)
                    writer = tf.python_io.TFRecordWriter(cur_tffilename)
                #print(img,mask)
                shape=img.shape
                #print(shape)
                img_data=img.tostring()
                mask_data=mask.tostring()
                example=tf.train.Example(features=tf.train.Features(
                    feature={
                    'shape': tf.train.Feature(int64_list = tf.train.Int64List(value=[shape[0],shape[1],shape[2]])),
                    'img_data':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_data])),
                    'mask_data':tf.train.Feature(bytes_list = tf.train.BytesList(value=[mask_data]))
                    }))
                writer.write(example.SerializeToString())
            
            except tf.errors.OutOfRangeError:
                print("end")
                break
    writer.close()
    return 0
