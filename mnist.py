#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size])+0.1)
    wxb = tf.matmul(inputs, W) + b
    if activation_function is None:
        output = wxb
    else:
        output = activation_function(wxb)
    return output

def accuracy(x, y):
    global pred
    y_pred = sess.run(pred, feed_dict={xs:x})
    count = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accrate = tf.reduce_mean(tf.cast(count, tf.float32))
    rate = sess.run(accrate, feed_dict={xs: x, ys: y})
    return rate

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

pred = add_layer(xs, 784, 10, tf.nn.softmax)
CE = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(CE)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
    if i%50==0:
        print(accuracy(mnist.test.images, mnist.test.labels))