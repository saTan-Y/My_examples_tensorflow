#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size])+0.1)
    wxb = tf.matmul(inputs, W) + b
    if activation_function is None:
        output = wxb
    else:
        output = activation_function(wxb)
    return output

x = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x.shape).astype(np.float32)
y = np.square(x) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(x, 1, 10, tf.nn.relu)
pred = add_layer(l1, 10, 1, None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred), reduction_indices=1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.scatter(x, y)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train, feed_dict={xs:x, ys:y})
    if i%50 == 0:
        print(sess.run(loss, feed_dict={ys: y}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        y_pred = sess.run(pred, feed_dict={xs: x})
        lines = ax.plot(x, y_pred, 'r-', lw=2)
        plt.pause(0.1)





