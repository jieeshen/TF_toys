#! /usr/bin/env python
#
#
#       Module: tf_reader
#
#       Description:
#      
#       Dependence:
#
#       Usage:
#

import tensorflow as tf

filename_queue = tf.train.string_input_producer(['test1.tfrecord'], num_epochs=None,
                                                shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'X': tf.FixedLenFeature([20000], tf.float32),  #
                                       'y': tf.FixedLenFeature([], tf.int64)}     #
                                   )
X_out = features['X']
y_out = features['y']

print(X_out)
print(y_out)

X_batch, y_batch = tf.train.shuffle_batch([X_out, y_out], batch_size=2,
                                          capacity=200, min_after_dequeue=100, num_threads=2)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
y_outputs = list()
for i in xrange(5):
    _X_batch, _y_batch = sess.run([X_batch, y_batch])
    print('** batch %d' % i)
    print('_X_batch:', _X_batch)
    print('_y_batch:', _y_batch)
    y_outputs.extend(_y_batch.tolist())
print(y_outputs)


coord.request_stop()
coord.join(threads)