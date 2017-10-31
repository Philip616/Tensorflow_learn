# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np

b = tf.Variable(tf.zeros([100]))
W = tf.Variable(tf.random_uniform([784,100],-1,1))
x = tf.placeholder(tf.float32)
relu = tf.nn.relu(tf.matmul(W,x)+b)
C = [...]
s = tf.Session()

for step in range (10):
    input = np.random.rand(100)
    result = s.run({x: input})
    print(step, result)
    