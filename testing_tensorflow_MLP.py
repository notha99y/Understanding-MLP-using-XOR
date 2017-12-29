import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import keras

from data_utils import generate_XOR_distribution
# Generating XOR
Input,Output = generate_XOR_distribution(1)

# Building mlp
n_input = 2
n_hidden = 10
n_output = 2

# Variables
X = tf.placeholder(tf.float32, shape = [None,n_input], name = "Input")
W1 = tf.get_variable(name = 'W1', shape=[n_input,n_hidden],
                     dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name = 'W2', shape=[n_hidden,n_output],
                     dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name = 'b1', shape = [n_hidden],
                    dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name = 'b2', shape = [n_output],
                    dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

# One hot encoding
Output = keras.utils.to_categorical(Output,num_classes=2)

# Operations
h1 = tf.nn.sigmoid(tf.add(tf.matmul(X,W1),b1), name = 'h1')
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(h1,W2),b2), name = 'Y_hat')

# Training
cost = tf.reduce_mean(tf.squared_difference(Y_hat,Output), name = 'cost')
train = tf.train.AdamOptimizer(learning_rate = 10e-3).minimize(cost)
initOP = tf.global_variables_initializer()

with tf.Session() as sess:
    tic = time.time()
    sess.run(initOP)
    epoch = 10000
    for i in range(epoch):
        sess.run(train,feed_dict={X:Input})
        if i%100 == 0:
            error = cost.eval(feed_dict={X:Input})
            toc = time.time()
            print("Epoch: {}, Error: {}, Time Taken: {}".format(i, error, toc-tic))

    print(sess.run(W1))
    print(sess.run(W2))
    print(sess.run(b1))
    print(sess.run(b2))
    saveFileName = 'weights//2_{}_2.csv'.format(n_hidden)
    with open(saveFileName,'w') as saveFile:
        saveFile.write(str(sess.run(W1))+"\n")
        saveFile.write("\n")
        saveFile.write(str(sess.run(W2)) + "\n")
        saveFile.write("\n")
        saveFile.write(str(sess.run(b1)) + "\n")
        saveFile.write("\n")
        saveFile.write(str(sess.run(b2)) + "\n")
    saveFile.close()
    print("File saved as: ",saveFileName)
