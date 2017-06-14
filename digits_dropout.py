#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def add_layer(inputs, in_size, out_size, layer_name, activation_function):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size])+0.1)
    wxb = tf.matmul(inputs, W) + b
    wxb = tf.nn.dropout(wxb, keep_prob)
    if activation_function is None:
        output = wxb
    else:
        output = activation_function(wxb)
    tf.summary.histogram(layer_name + '/outputs', output)
    return output

digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'l1', tf.nn.tanh)
pred = add_layer(l1, 50, 10, 'l2', tf.nn.softmax)
CE = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred), reduction_indices=[1]))
tf.summary.scalar('loss', CE)
train = tf.train.GradientDescentOptimizer(0.1).minimize(CE)

sess = tf.Session()
summary = tf.summary.merge_all()
train_log = tf.summary.FileWriter('log/train', sess.graph)
test_log = tf.summary.FileWriter('log/test', sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    # batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i%50==0:
        train_res = sess.run(summary, feed_dict={xs: x_train, ys: y_train, keep_prob:0.5})
        test_res = sess.run(summary, feed_dict={xs: x_test, ys: y_test, keep_prob: 0.5})
        # train_log = tf.summary.FileWriter.add_summary(train_res)
        # test_log = tf.summary.FileWriter.add_summary(test_res)
        # train_writer.add_summary(train_result, i)
        # test_writer.add_summary(test_result, i)