#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

if __name__ == '__main__':
    a1 = tf.placeholder(tf.float32)
    a2 = tf.placeholder(tf.float32)

    b = tf.multiply(a1, a2)

    with tf.Session() as sess:
        print(sess.run(b, feed_dict={a1:[3], a2:[4]}))