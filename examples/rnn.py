# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn

# ======================================================================= #

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("Download Done!\n")

# parameters
learning_rate  = 1e-3
training_iters = 100000
batch_size     = 50
display_step   = 100
logs_path      = 'TensorBoard/rnn'
model_path     = "Model/rnn_model.ckpt"

# network Parameters
n_input   = 28  # input data (img shape: 28*28)
n_steps   = 28  # timesteps
n_hidden  = 150 # hidden layer num of features
n_classes = 10  # total classes (0-9 digits)

# tf graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input], name = 'InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'LabelData')

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
    #return tf.matmul(outputs[-1], weights['out']) + biases['out']

# ======================================================================= #

with tf.name_scope('Model'):
    # build model
    pred = RNN(x, weights, biases)

# define loss and optimizer
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred))

with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# evaluate model
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)

merged_summary_op = tf.summary.merge_all()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# ======================================================================= #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

step = 1

while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    # run optimization op (backprop)
    sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

    if step % display_step == 0:
        # calculate batch loss and accuracy
        loss, acc, summary = sess.run([cost, accuracy, merged_summary_op], \
                                      feed_dict = {x: batch_x, y: batch_y})

        summary_writer.add_summary(summary, step * batch_size)

        print("Iter               : %d  " %(step * batch_size))
        print("Minibatch Loss     : %.2f" %(loss))
        print("Training  Accuracy : %.2f" %(acc), "\n")

    step += 1

print("Optimization Finished!")

# Save model weights to disk
save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

sess.close()

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

print("Run the command line : ")
print("--> tensorboard --logdir=TensorBoard/rnn")
print("Then open localhost:6006 into your web browser")

sess.close()
