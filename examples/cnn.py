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
logs_path      = 'TensorBoard/cnn'
model_path     = "Model/cnn_model.ckpt"

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

# ======================================================================= #

weights = {
    # 5 x 5 conv,  1 input , 32 outputs
    'W_conv1': weight_variable([5, 5, 1, 32], name = 'W_conv1'),
    # 5 x 5 conv, 32 inputs, 64 outputs
    'W_conv2': weight_variable([5, 5, 32, 64], name = 'W_conv2'),
    # 7 * 7 * 64 inputs, 1024 outputs
    'W_fc1': weight_variable([7 * 7 * 64, 1024], name = 'W_fc1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out'  : weight_variable([1024, n_classes], name = 'out')
}

biases = {
    'b_conv1': bias_variable([32], name = 'b_conv1'),
    'b_conv2': bias_variable([64], name = 'b_conv2'),
    'b_fc1': bias_variable([1024], name = 'b_fc1'),
    'out'  : bias_variable([n_classes], name = 'out')
}

# ======================================================================= #

# create model
def CNN(x, weights, biases, dropout):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv layer-1
    h_conv1 = conv2d(x, weights['W_conv1'], biases['b_conv1'])
    h_pool1 = max_pool(h_conv1, k = 2)

    # conv layer-2
    h_conv2 = conv2d(h_pool1, weights['W_conv2'], biases['b_conv2'])
    h_pool2 = max_pool(h_conv2, k = 2)

    # full connection
    # reshape conv2 output to fit fully connected layer input
    h_pool2_flat = tf.reshape(h_pool2, [-1, weights['W_fc1'].get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, weights['W_fc1']), biases['b_fc1']))

    # apply dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    # output layer, class prediction
    out = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, weights['out']), biases['out']))
    #out = tf.add(tf.matmul(h_fc1_drop, weights['out']), biases['out'])
    return out

# ======================================================================= #

with tf.name_scope('Model'):
    # build model
    pred = CNN(x, weights, biases, keep_prob)

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
print("--> tensorboard --logdir=TensorBoard/cnn")
print("Then open localhost:6006 into your web browser")

sess.close()
