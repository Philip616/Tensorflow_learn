<<<<<<< HEAD
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
=======
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:22:21 2017

@author: hyps4
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

#loss Function
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
>>>>>>> 8b4128a0ed4e33e5b3f702996839cae85b2443c6
