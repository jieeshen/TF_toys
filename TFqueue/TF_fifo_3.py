#! /usr/bin/env python
#
#
#       Module: TF_fifo
#
#       Description:
#      
#       Dependence:
#
#       Usage:
#

import tensorflow as tf
import numpy as np
import pandas as pd
from pybase.support import lly_tempfile
from time import time
import os, shutil

N_SAMPES = 10
N_FEATURE = 3
N_THREADS = 4
BATCHSIZE = 3
FILELIMIT = 5


def np2tfrecord(tfrfile, x, y):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    temp_path = lly_tempfile.NamedTemporaryFile(delete=False).name + "_DNN/"
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
    else:
        shutil.rmtree(temp_path)
        os.mkdir(temp_path)

    filenames = []
    l = len(y)
    idx = 0
    for f in xrange(l / FILELIMIT + 1):
        fileidx = f + 1
        tfrfile_name = temp_path + tfrfile + str(fileidx)
        if idx < l:
            writer = tf.python_io.TFRecordWriter(tfrfile_name)
        else:
            break
        singlesize = 0
        while idx < l and singlesize < FILELIMIT:
            example = tf.train.Example(features=tf.train.Features(feature={
                'x': _float_feature(x[idx]),
                'y': _int64_feature(y[idx])
            }))
            writer.write(example.SerializeToString())
            idx += 1
            singlesize += 1

        writer.close()
        filenames.append(tfrfile_name)

    # writer.close()
    print filenames
    return filenames


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'x': tf.FixedLenFeature([N_FEATURE], tf.float32),
            'y': tf.FixedLenFeature([], tf.int64)
        })

    x_single = features['x']
    # x_single.set_shape([N_FEATURE])
    y_single = features['y']
    return x_single, y_single


def input_pipeline(filenames, batch_size, num_epochs=None, num_features=None):
    '''num_features := width * height for 2D image'''
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=None, shuffle=True)
    example, label = read_and_decode(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 80
    capacity = min_after_dequeue + 3 * batch_size
    print "$$$$$$$$$$$$$$$$"
    print example, label
    # example_batch, label_batch = tf.train.shuffle_batch(
    #         [example, label], batch_size=batch_size, capacity=capacity,
    #         min_after_dequeue=min_after_dequeue)
    example_batch, label_batch = tf.train.batch([example, label], batch_size=BATCHSIZE, num_threads=4)
    return example_batch, label_batch


def read_test(filenames):
    '''example usage to read batches of records from TFRcords files'''
    # filenames = ['test.tfrecords']
    example_batch, label_batch = input_pipeline(filenames, batch_size=BATCHSIZE,
                                                num_epochs=None, num_features=N_FEATURE)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in xrange(3):
            for step in xrange(N_SAMPES / BATCHSIZE + (N_SAMPES % BATCHSIZE <> 0)):
                print(sess.run(label_batch))
                images, labels = sess.run([example_batch, label_batch])
                print(images, labels)
            print("epoch", epoch)
        coord.request_stop()
        # # try:
        # if 1:
        #     idx = 0
        #     # while not coord.should_stop():
        #     while idx<100:
        #         # Run training steps or whatever
        #         images, labels = sess.run([example_batch, label_batch])
        #         print(images, labels)
        #         idx = idx + 1
        #         if idx > 2: break
        # # except tf.errors.OutOfRangeError:
        # #     print('Done training -- epoch limit reached')
        # # finally:
        #     # When done, ask the threads to stop.
        #     coord.request_stop()

        coord.join(threads)


def check_ids(Xids, Yids):
    assert len(Xids) == len(Yids), 'X labels and Y labels do not match'
    l = max(len(Xids), len(Yids))
    for index in xrange(l):
        assert Yids[index] == Xids[index], 'X labels and Y labels do not match'


def main():
    # all_x = 10 * np.random.randn(N_SAMPES, N_FEATURE) + 1
    # all_y = np.random.randint(0, 2, size=N_SAMPES)
    # print [all_x, all_y]
    # x_df = pd.DataFrame(all_x)
    # y_df = pd.DataFrame(all_y)
    # x_df.to_csv("x.csv", sep=" ", index_label="id")
    # y_df.to_csv("y.csv", sep=" ", index_label="id")
    tos = time()
    # tempx = pd.read_csv("/node/scratch/20929706.1.all.ia.q/tmpSilS78_DNN/lsn", sep=" ", nrows=3)
    tempy = pd.read_csv("/lrlhps/scratch/c225797/DeepLearning/LLYMolNet/data/all.2d", sep=" ", nrows=3)
    ytitle = list(tempy)[1:100]
    print ytitle
    tempy1 = pd.read_table("/lrlhps/scratch/c225797/DeepLearning/LLYMolNet/data/all.2d", sep=" ", usecols=ytitle[1:100])
    print tempy1,time()-tos


    # print time() - tos
    # chunksize = 5000
    # xdfs = pd.read_csv("/node/scratch/20929706.1.all.ia.q/tmpSilS78_DNN/lsn", sep=" ", chunksize=chunksize,
    #                    iterator=True, dtype={xtitle[0]: str})
    # ydfs = pd.read_csv("/node/scratch/20929706.1.all.ia.q/tmpSilS78_DNN/new.y", sep=" ", chunksize=chunksize,
    #                    iterator=True, dtype={xtitle[0]: str})
    # ydf = pd.read_csv("/node/scratch/20929706.1.all.ia.q/tmpSilS78_DNN/new.y", sep=" ", dtype={xtitle[0]: str})
    # yids=np.asarray(ydf.iloc[:,0])
    # y_values=np.asarray(ydf.iloc[:,1:])
    # print len(ydf)
    # print yids
    # print ydf
    # print y_values
    # i = 0
    # print range(0,len(ydf),100000)
    # for i in range(0,len(ydf),100000):
    #     try:
    #         print "$$$$$$$$$$$$$$$$$"
    #         xdf_c = xdfs.get_chunk(100000)
    #         print xdf_c.dtypes
    #         # print "###################"
    #         ydf_c = ydf.iloc[i:i+100000,:]
    #         print yids[i:i+100000],y_values[i:i+100000,:]
    #         check_ids(np.asarray(xdf_c.iloc[:, 0]), yids[i:i+100000])
    #     except StopIteration:
    #         break
    # print "done!!!"



    # if len(y_df)>4:
    #     y_df.drop(y_df.index[range(4)],inplace=True)
    # print y_df

    # filenames=np2tfrecord("training.tfr", all_x, all_y)

    # read_test(filenames)

    # queue = tf.train.string_input_producer(["test.tfrecords"], num_epochs=10, shuffle=False)
    # x, y = read_and_decode(queue)
    # print x, y
    # # dataset=tf.data.Dataset.from_tensor_slices((all_x,all_y))
    # # tf.random_shuffle([all_x,all_y])
    #
    # # queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32,tf.int32],shapes=[[8],[]])
    # # queue = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=0, dtypes=[tf.float32,tf.int32],shapes=[[4],[]])
    # # queue = tf.train.slice_input_producer([all_x,all_y],shuffle=True)
    # # x=queue[0]
    # # y=queue[1]
    # # x=tf.placeholder(tf.float32,shape=(N_SAMPES,10000))
    # # y=tf.placeholder(tf.float32,shape=[N_SAMPES,1])
    # # x=tf.convert_to_tensor(all_x)
    # # y=tf.convert_to_tensor(all_y)
    # min_after_dequeue = 1000
    # capacity = min_after_dequeue + 3 * BATCHSIZE
    # x_batch, y_batch = tf.train.shuffle_batch([x, y], batch_size=BATCHSIZE, num_threads=N_THREADS, capacity=capacity,
    #                                           min_after_dequeue=min_after_dequeue)
    # # enqueue_op = queue.enqueue_many([all_x,all_y])
    # # data_sample, label_sample = queue.dequeue_many(2)
    # # label_sample=tf.Print(label_sample, label_sample=[queue.size()],message="left in queue: ")
    #
    # # qr=tf.train.QueueRunner(queue, [enqueue_op] * N_THREADS)
    # # tf.train.add_queue_runner(qr)
    #
    # with tf.Session() as sess:
    #     # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     # sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
    #     for epoch in xrange(3):
    #         for step in xrange(N_SAMPES / BATCHSIZE + (N_SAMPES % BATCHSIZE <> 0)):
    #             # if coord.should_stop():
    #             #     break
    #             one_data, one_label = sess.run([x_batch, y_batch])
    #             print step,
    #             print(one_data),
    #             print(one_label)
    #     coord.request_stop()
    #     coord.join(enqueue_threads)


if __name__ == '__main__':
    main()
