#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

if __name__ == '__main__':
    state = tf.Variable(0, name='test')
    one = tf.constant(1)
    new_state = tf.add(state, one)
    update = tf.assign(state, new_state)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(3):
            sess.run(update)
            print(sess.run(state))