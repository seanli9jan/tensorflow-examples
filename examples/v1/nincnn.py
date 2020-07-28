# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

# ======================================================================= #

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("Download Done!\n")

# parameters
learning_rate  = 1e-3
training_iters = 100000
batch_size     = 50
display_step   = 100
logs_path      = 'TensorBoard/cnn_nin'
model_path     = "Model/cnn_nin_model.ckpt"

# network parameters
n_input   = 784  # input data (img shape: 28 * 28)
n_classes = 10   # total classes (0-9 digits)
dropout   = 0.75 # dropout, probability to keep units

# tf graph input
x = tf.placeholder(tf.float32, [None, n_input], name = 'InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'LabelData')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# ======================================================================= #

# create some wrappers for simplicity
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W,                               \
                     strides = [1, strides, strides, 1], \
                     padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool(x, k = 2):
    return tf.nn.max_pool(x,                      \
                          ksize   = [1, k, k, 1], \
                          strides = [1, k, k, 1], \
                          padding = 'SAME')
def avg_pool(x, k = 2):
    return tf.nn.avg_pool(x,                      \
                          ksize   = [1, k, k, 1], \
                          strides = [1, k, k, 1], \
                          padding = 'SAME')

# ======================================================================= #

weights = {
    # 5 x 5 conv,  1 input , 32 outputs
    'W_conv1': weight_variable([5, 5, 1, 32], name = 'W_conv1'),
    # mlp conv, 32 input , 32 outputs
    'W_cccp1': weight_variable([1, 1, 32, 32], name = 'W_cccp1'),
    # 5 x 5 conv, 32 inputs, 64 outputs
    'W_conv2': weight_variable([5, 5, 32, 64], name = 'W_conv2'),
    # mlp conv, 64 input , 64 outputs
    'W_cccp2': weight_variable([1, 1, 64, 64], name = 'W_cccp2'),
    # 5 x 5 conv, 64 inputs, 128 outputs
    'W_conv3': weight_variable([5, 5, 64, 128], name = 'W_conv3'),
    # mlp conv, 128 input , 10 outputs
    'W_cccp3': weight_variable([1, 1, 128, n_classes], name = 'W_cccp3')
}

biases = {
    'b_conv1': bias_variable([32], name = 'b_conv1'),
    'b_cccp1': bias_variable([32], name = 'b_cccp1'),
    'b_conv2': bias_variable([64], name = 'b_conv2'),
    'b_cccp2': bias_variable([64], name = 'b_cccp2'),
    'b_conv3': bias_variable([128], name = 'b_conv3'),
    'b_cccp3': bias_variable([n_classes], name = 'b_cccp3')
}

# ======================================================================= #

# create model
def CNN_NIN(x, weights, biases, dropout):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv layer-1
    h_conv1 = conv2d(x, weights['W_conv1'], biases['b_conv1'])
    h_cccp1 = conv2d(h_conv1, weights['W_cccp1'], biases['b_cccp1'])
    h_pool1 = max_pool(h_cccp1, k = 2)
    #h_pool1_drop = tf.nn.dropout(h_pool1, dropout)

    # conv layer-2
    h_conv2 = conv2d(h_pool1, weights['W_conv2'], biases['b_conv2'])
    h_cccp2 = conv2d(h_conv2, weights['W_cccp2'], biases['b_cccp2'])
    h_pool2 = max_pool(h_cccp2, k = 2)
    #h_pool2_drop = tf.nn.dropout(h_pool2, dropout)

    # conv layer-3
    h_conv3 = conv2d(h_pool2, weights['W_conv3'], biases['b_conv3'])
    h_cccp3 = conv2d(h_conv3, weights['W_cccp3'], biases['b_cccp3'])
    h_pool3 = avg_pool(h_cccp3, k = 7)

    h_pool3_flat = tf.reshape(h_pool3, [-1, n_classes])
    out = tf.nn.softmax(h_pool3_flat)

    return out

# ======================================================================= #

with tf.name_scope('Model'):
    # build model
    pred = CNN_NIN(x, weights, biases, keep_prob)

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
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

step = 1

while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

    if step % display_step == 0:
        # calculate batch loss and accuracy
        loss, acc, summary = sess.run([cost, accuracy, merged_summary_op], \
                                      feed_dict = {x: batch_x, y: batch_y, \
                                      keep_prob: 1.})

        summary_writer.add_summary(summary, step * batch_size)

        print("Iter              : %d  " %(step * batch_size))
        print("Minibatch Loss    : %.2f" %(loss))
        print("Training  Accuacy : %.2f" %(acc), "\n")

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
                                x: mnist.test.images,              \
                                y: mnist.test.labels,              \
                                keep_prob: 1.})), "\n")

print("Run the command line : ")
print("--> tensorboard --logdir=TensorBoard/cnn_nin")
print("Then open localhost:6006 into your web browser")

sess.close()
