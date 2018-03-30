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
N_SAMPES=10
N_THREADS=10

def main():
    all_x = 10 * np.random.randn(N_SAMPES,4)+1
    all_y = np.random.randint(0,2,size=N_SAMPES)
    print [all_x, all_y]

    # tf.random_shuffle([all_x,all_y])

    # queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32,tf.int32],shapes=[[8],[]])
    queue = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=0, dtypes=[tf.float32,tf.int32],shapes=[[4],[]])
    enqueue_op = queue.enqueue_many([all_x,all_y])
    data_sample, label_sample = queue.dequeue_many(2)
    # label_sample=tf.Print(label_sample, label_sample=[queue.size()],message="left in queue: ")

    qr=tf.train.QueueRunner(queue, [enqueue_op] * N_THREADS)
    tf.train.add_queue_runner(qr)

    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        for epoch in xrange(3):
            for step in xrange(10/2):
                # if coord.should_stop():
                #     break
                one_data, one_label = sess.run([data_sample,label_sample])
                print step,
                print(one_data),
                print(one_label)
        coord.request_stop()
        coord.join(enqueue_threads)



if __name__ == '__main__':
    main()