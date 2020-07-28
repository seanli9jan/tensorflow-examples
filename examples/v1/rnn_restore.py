# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn

# ======================================================================= #

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("Download Done!\n")

# parameters
model_path     = "Model/rnn_model.ckpt"

# network Parameters
n_input   = 28  # input data (img shape: 28*28)
n_steps   = 28  # timesteps
n_hidden  = 150 # hidden layer num of features
n_classes = 10  # total classes (0-9 digits)

# tf graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# ======================================================================= #

# create some wrappers for simplicity
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

# ======================================================================= #

weights = {
    'out': weight_variable([n_hidden, n_classes], name = 'Weights')
}

biases = {
    'out': bias_variable([n_classes], name = 'Bias')
}

# ======================================================================= #

# create model
def RNN(x, weights, biases):
    # unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

    # get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)

    # linear activation, using rnn inner loop last output
    return tf.nn.softmax(tf.add(tf.matmul(outputs[-1], weights['out']),biases['out']))

# ======================================================================= #

# build model
pred = RNN(x, weights, biases)

# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# ======================================================================= #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Restore model weights from previously saved model
saver.restore(sess, model_path)
print("Model restored from file: %s" % model_path)

# accuacy on test
print("Testing Accuracy : %.2f" %(sess.run(accuracy, feed_dict = { \
                                x: mnist.test.images.reshape       \
                                ((-1, n_steps, n_input)),          \
                                y: mnist.test.labels})), "\n")

sess.close()
