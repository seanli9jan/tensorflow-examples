# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

class layers:
    bias_initializer = tf.constant_initializer(0.1)
    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1)
 
    @classmethod
    def conv2d(self, inputs, filters, kernel_size, strides = (1, 1), \
               padding = 'same', activation = None, name = None):
        return tf.layers.conv2d(inputs, filters, kernel_size, strides = strides, \
                                padding = padding, activation = activation,      \
                                kernel_initializer = self.kernel_initializer,    \
                                bias_initializer = self.bias_initializer, name = name)

    @classmethod
    def max_pool2d(self, inputs, pool_size = 2, strides = 2, padding = 'same', \
                   name = None):
        return tf.layers.max_pooling2d(inputs, pool_size, strides, padding = padding, \
                                       name = name)

    @classmethod
    def dense(self, inputs, units, activation = None, name = None):
        return tf.layers.dense(inputs, units, activation = activation,       \
                               kernel_initializer = self.kernel_initializer, \
                               bias_initializer = self.bias_initializer, name = name)
