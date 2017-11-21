# -*- coding: utf-8 -*-

from __future__ import print_function
from tensorflow import keras

import tensorflow as tf

# ===================================================================== #

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("Download Done!\n")

# parameters
learning_rate  = 1e-3
training_iters = 100000
batch_size     = 50
display_step   = 100
logs_path      = 'TensorBoard/cnn'
model_path     = "Model/cnn_model.ckpt"

# network parameters
n_input   = 784  # input data (img shape: 28 * 28)
n_classes = 10   # total classes (0-9 digits)
dropout   = 0.25 # dropout, probability to keep units

# Returns the learning phase flag (0 = test, 1 = train)
learning_phase = keras.backend.learning_phase()

# tf graph input
x = tf.placeholder(tf.float32, [None, n_input], name = 'InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'LabelData')

# ======================================================================= #
# create model
def CNN(x):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv layer-1
    h_conv1 = keras.layers.Conv2D(32, (5, 5), padding = 'same', activation = 'relu', input_shape = x.shape)(x)
    h_conv1 = keras.layers.Conv2D(32, (1, 1), padding = 'same', activation = 'relu')(h_conv1)
    h_conv1 = keras.layers.Conv2D(32, (1, 1), padding = 'same', activation = 'relu')(h_conv1)
    h_pool1 = keras.layers.MaxPooling2D((3, 3), strides = 2, padding = 'same')(h_conv1)

    # conv layer-2
    h_conv2 = keras.layers.Conv2D(64, (5, 5), padding = 'same', activation = 'relu')(h_pool1)
    h_conv2 = keras.layers.Conv2D(64, (1, 1), padding = 'same', activation = 'relu')(h_conv2)
    h_conv2 = keras.layers.Conv2D(64, (1, 1), padding = 'same', activation = 'relu')(h_conv2)
    h_pool2 = keras.layers.MaxPooling2D((3, 3), strides = 2, padding = 'same')(h_conv2)

    # full connection
    # reshape conv2 output to fit fully connected layer input
    h_pool2_flat = keras.layers.Flatten()(h_pool2)
    h_fc1 = keras.layers.Dense(1024, activation = 'relu')(h_pool2_flat)

    # apply dropout
    h_fc1_drop = keras.layers.Dropout(dropout)(h_fc1)
    #h_fc1_drop = tf.nn.dropout(h_fc1, 1-dropout)

    # output layer, class prediction
    out = keras.layers.Dense(n_classes, activation = 'softmax')(h_fc1_drop)
    return out

# ======================================================================= #

with tf.name_scope('Model'):
    # build model
    pred = CNN(x)

# define loss and optimizer
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))

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
keras.backend.set_session(sess)

# Initialization all parameters
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

step = 1

while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, learning_phase: 1})

    if step % display_step == 0:
        # calculate batch loss and accuracy
        loss, acc, summary = sess.run([cost, accuracy, merged_summary_op], \
                                      feed_dict = {x: batch_x, y: batch_y, \
                                      learning_phase: 0})

        summary_writer.add_summary(summary, step * batch_size)

        print("Iter              : %d  " %(step * batch_size))
        print("Minibatch Loss    : %.2f" %(loss))
        print("Training  Accuacy : %.2f" %(acc), "\n")

    step += 1

print("Optimization Finished!")

# Save model weights to disk
save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

# Initialization all parameters
sess.run(tf.global_variables_initializer())

# Restore model weights from previously saved model
saver.restore(sess, model_path)
print("Model restored from file: %s" % model_path)

# accuacy on test
print("Testing Accuracy : %.2f" %(sess.run(accuracy, feed_dict = { \
                                x: mnist.test.images,              \
                                y: mnist.test.labels,              \
                                learning_phase: 0})), "\n")

print("Run the command line : ")
print("--> tensorboard --logdir=TensorBoard/cnn")
print("Then open localhost:6006 into your web browser")

keras.backend.clear_session()
sess.close()
