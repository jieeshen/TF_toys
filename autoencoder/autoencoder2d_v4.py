#! /usr/bin/env python
#
#
#
#   Updated: 10/26/2017
#           remove tensorboard and bias
#
#
#
#
#   Updated: 10/26/2017
#           Trying using functions to create the tensorflow model, including more options,
#               i.e.    activation function
#                       batch normal
#                       dropout
#
#       Description: This is used for chemical unsupervised clustering/visulization
#      
#       Dependence:
#
#       Usage:
#
from optparse import OptionParser
import os,sys,subprocess,math,shutil
from time import sleep, time
import numpy as np
import pandas as pd
import tensorflow as tf
from pybase.support import lly_tempfile

MAX_JOB_TIME = 20

def run(*args):
    cmd = ' '.join(args)
    #print ("Running ", cmd)
    # print ("BABEL DIR=", BABEL_DIR)
    outlog=open("stdout.log","w")
    errlog=open("stderr.log","w")
    pipe = subprocess.Popen(cmd, stdout=outlog, stderr=errlog, shell=True)
    #pipe = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
    #pipe = subprocess.Popen(cmd,shell=True)
    deadline = time() + MAX_JOB_TIME * 60
    while True:
        result = pipe.poll()
        if result == None:
            if time() > deadline:
                pipe.kill()
                raise RuntimeError("Job '%s' not finished after %d minutes" % (cmd, MAX_JOB_TIME))
            else:
                sleep(1)
        else:
            outlog.close()
            errlog.close()
            return result



def docterm2matrix(dfinfilename):
    dfin=pd.read_csv(dfinfilename,sep="\t",low_memory=False)
    # print dfin
    df=dfin.pivot(index="doc",columns='term',values='frequency').fillna(0)
    # df.to_csv("/lrlhps/users/c225797/TKA/TopicModeling/DocTerm_Freq_matrix.csv")
    # print df
    dfnp=np.asarray(df)
    return dfnp


def smi2desc(smifile,fptype,nbits):
    runningdir=os.getcwd()
    temp_path=lly_tempfile.NamedTemporaryFile(delete=True).name+"_DNN"
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
    else:
        shutil.rmtree(temp_path)
        os.mkdir(temp_path)
    # shutil.copy(smifile,temp_path)
    os.chdir(temp_path)
    run("gfp_make.sh",fptype, runningdir+"/"+smifile, "> smi.gfp")
    run("gfp_to_descriptors_multiple.sh -d ",nbits,"smi.gfp>smi.desc")
    descdf=pd.read_csv("smi.desc",sep=" ")
    os.chdir(runningdir)
    return descdf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial,tf.float32)

def create_hid_layer(num_in,num_out,data,activation,keep_prob,batchnorm):
    W_fc = weight_variable([num_in, num_out])
    b_fc = bias_variable([num_out])
    z_fc = tf.matmul(data, W_fc) + b_fc
    if batchnorm:
        batch_mean,batch_var=tf.nn.moments(z_fc,[0])
        beta=tf.Variable(tf.zeros([num_out]))
        scale=tf.Variable(tf.ones([num_out]))
        z_fc_bn = tf.nn.batch_normalization(z_fc,batch_mean,batch_var,beta,scale,1e-4)
    else:
        z_fc_bn=z_fc
    if activation=="relu":
        h_fc = tf.nn.relu(z_fc_bn)
    elif activation=="sigmoid":
        h_fc = tf.nn.sigmoid(z_fc_bn)
    elif activation=="tanh":
        h_fc = tf.nn.tanh(z_fc_bn)
    else:
        h_fc=z_fc_bn
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
    return h_fc_drop

def main():
    parser = OptionParser(
        description='This is to build Deep AutoEncoder to visulize the high dimentional data in 2D plots'
                    'For more info or enhancements please contact Jie Shen (jshen@lilly.com)')

    parser.add_option('-i', dest='inmatrix', help='The input numpy matrix, the first colunm is the doc ID', action='store', default=False)
    parser.add_option('-o', dest='ftout', help='The output file of the feature learned by the autoencoder', action='store', default=False)
    # parser.add_option('--vsmi', dest='vsmi', help='The SMILES file of the validation/development set', action='store', default=False)
    # parser.add_option('--vy', dest='vy', help='TThe Y file of the validation/development set', action='store', default=False)
    # parser.add_option('--fp', dest='fptype', help='The GFP type, e.g. -CATS13 -MAP9:ACHP1 -RS -PSA -CLOGP', action='store', default=False)
    # parser.add_option('-n', dest='nbits', help='The GFP bits for descriptors,default: 1024', action='store', default="1024")
    parser.add_option('-d', dest='mdir', help='The name stem of checkpoint files, for generating', action='store', default=False)
    parser.add_option('-c', dest='ckpt', help='The name stem of checkpoint files, for restoring', action='store', default=False)
    parser.add_option('-e', dest='nepoch', help='The total epoch number', action='store', default="1000")
    # parser.add_option('-o', dest='outfile', help='The output file of the training log', action='store', default=False)
    parser.add_option('-k', dest='keepprop', help='The dropout keeping rate', action='store', default="0.5")
    parser.add_option('-l', dest='lr', help='The learning rate for AdamOptimizer', action='store', default="0.0001")
    parser.add_option('-b', dest='beta', help='The beta for regularizers, 0 means no regularizer', action='store', default="0")
    parser.add_option('-s', dest='batchsize', help='The batch size for minibatach training', action='store', default="128")
    parser.add_option('--bn', dest='batchnorm', help='Wether to use batch normalization', action='store', default="False")
    parser.add_option('-a', dest='actv', help='The activation function (sigmoid or relu or tanh or noact)', action='store', default="Relu")
    # parser.add_option('--tyhat', dest='tyhat', help='The Y hat file of the training set', action='store', default="train.yhat")
    # parser.add_option('--vyhat', dest='vyhat', help='TThe Y hat file of the validation/development set', action='store', default="valid.yhat")
    (args, ignore) = parser.parse_args()
    # create the working directory if not existed
    if args.inmatrix == False or args.ftout == False or args.mdir == False:
        parser.print_help()
        sys.exit(-1)

    beta=float(args.beta)
    batchsize=int(args.batchsize)
    lr=float(args.lr)
    nepoch = int(args.nepoch)
    kprop = float(args.keepprop)
    batchnorm=args.batchnorm
    actv=args.actv
    df=pd.read_csv(args.inmatrix,sep=" ")
    trainX=np.asarray(df.drop(df.columns[[0]],axis=1))

    n_x_t=trainX.shape[0]
    print(trainX.shape)
    n_x=trainX.shape[1]
    # print trainX
    x=tf.placeholder(tf.float32,shape=[None,n_x])
    xhat=tf.placeholder(tf.float32,shape=[None,n_x])
    keep_prob = tf.placeholder(tf.float32)

#Build the encoder
    # First Layer
    h1=create_hid_layer(n_x,4096,x,actv,keep_prob,batchnorm)

    # Second Layer
    h2=create_hid_layer(4096,2048,h1,actv,keep_prob,batchnorm)

    # Third Layer
    h3=create_hid_layer(2048,512,h2,actv,keep_prob,batchnorm)

    # 4th Layer
    h4=create_hid_layer(512,128,h3,actv,keep_prob,batchnorm)

    #Code Layer
    f_2=create_hid_layer(128,2,h4,"sigmoid",1,False)

# Reverse
    #decoder_Layer
    f2_t=create_hid_layer(2,128,f_2,actv,keep_prob,batchnorm)

    #4th layer
    h4_t=create_hid_layer(128,512,f2_t,actv,keep_prob,batchnorm)
    #Third layer
    h3_t=create_hid_layer(512,2048,h4_t,actv,keep_prob,batchnorm)

    #second layer
    h2_t=create_hid_layer(2048,4096,h3_t,actv,keep_prob,batchnorm)

    # First Layer
    xhat=create_hid_layer(4096,n_x,h2_t,"noact",1,"False")


# regularizer
#     regularizers=tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(W_fc3)+tf.nn.l2_loss(W_fc4)+\
#                  tf.nn.l2_loss(W_fc1_t)+tf.nn.l2_loss(W_fc2_t)+tf.nn.l2_loss(W_fc3_t)+tf.nn.l2_loss(W_fc4_t)


    #cost
    cost=tf.reduce_sum(tf.pow(x-xhat,2)) #+beta*regularizers)

    #rms

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))'
    # tensorboard
    # tf.summary.scalar("cost", cost)
    # summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if args.ckpt:
            saver.restore(sess,args.ckpt)
        else:
            sess.run(tf.global_variables_initializer())
        for epoch in range(nepoch+1):
            indices=np.random.permutation(n_x_t)
            batches=[indices[i:i+batchsize] for i in range(0,len(indices),batchsize)]
            tos=time()
            # summary_writer = tf.summary.FileWriter("./TensorBoard", graph=tf.get_default_graph())
            for b in range(len(batches)):
                loss = sess.run(train_step,feed_dict={x: trainX[batches[b]], keep_prob: kprop})
                # _, loss, summary = (sess.run([train_step, cost, summary_op],
                #                          feed_dict={x: trainX[batches[b]], keep_prob: kprop}))
                # summary_writer.add_summary(summary, epoch)
            toe = time()
            if (epoch) % 10 ==0:
                train_loss = sess.run(cost, feed_dict={x: trainX, keep_prob: 1.0})
                print('epoch %d, training loss for all data %f' % (epoch, train_loss)),
                print('epoch %d takes %f min' %(epoch,(toe-tos)/60.))
                saver.save(sess, args.mdir, global_step=epoch)
                featureout=sess.run(f_2, feed_dict={x: trainX, keep_prob: 1.0})
                ft_df = pd.DataFrame(featureout)
                ft_df_new=pd.concat([df[[0]],ft_df],axis=1)
                ft_df_new.to_csv("out_"+str(epoch)+".csv", sep=" ",index=False)
        featureout=sess.run(f_2, feed_dict={x: trainX, keep_prob: 1.0})
        ft_df = pd.DataFrame(featureout)
        ft_df_new=pd.concat([df[[0]],ft_df],axis=1)
        ft_df_new.to_csv(args.ftout, sep=" ",index=False)
        # print('validation rms %g' % math.sqrt(valid_rms))



if __name__ == '__main__':
    main()