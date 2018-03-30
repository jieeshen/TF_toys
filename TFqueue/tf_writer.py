#! /usr/bin/env python
#
#
#       Module: tf_writer
#
#       Description:
#      
#       Dependence:
#
#       Usage:
#
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
# from tqdm import tqdm
N_SAMPES = 100000
N_FEATURE = 20000
N_THREADS = 2
BATCHSIZE = 2

all_x = 10 * np.random.randn(N_SAMPES, N_FEATURE) + 1
all_y = np.random.randint(0, 2, size=N_SAMPES)


writer1 = tf.python_io.TFRecordWriter('test1.tfrecord')
writer2 = tf.python_io.TFRecordWriter('test2.tfrecord')


# X = np.arange(0, 100).reshape([50, -1]).astype(np.float32)
# y = np.arange(50)

for i in xrange(len(all_y)):
    if i >= len(all_y) / 2:
        writer = writer2
    else:
        writer = writer1
    X_sample = all_x[i].tolist()
    y_sample = all_y[i]

    example = tf.train.Example(
        features=tf.train.Features(
            feature={'X': tf.train.Feature(float_list=tf.train.FloatList(value=X_sample)),
                     'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_sample]))}))

    serialized = example.SerializeToString()
    writer.write(serialized)

print('Finished.')
writer1.close()
writer2.close()
