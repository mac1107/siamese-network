import tensorflow as tf

import numpy as np


def siamese_loss(out1, out2, y, Q=5):
    Q = tf.constant(Q, name="Q", dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2), 1))
    pos = tf.multiply(tf.multiply(y, 2 / Q), tf.square(E_w))
    neg = tf.multiply(tf.multiply(1 - y, 2 * Q), tf.exp(-2.77 / Q * E_w))
    loss = pos + neg
    loss = tf.reduce_mean(loss)
    return loss


def siamese(inputs, keep_prob):
    # conv1+relu1
    with tf.name_scope('conv1') as scope:
        a = tf.random_normal_initializer
        # w1 = tf.Variable(tf.initializers.TruncatedNormal(stddev=0.05)(shape=[3, 3, 1, 32]), name='w1')
        w1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], stddev=0.05), name='w1')
        b1 = tf.Variable(tf.zeros(32), name='b1')
        conv1 = tf.nn.conv2d(inputs, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
    with tf.name_scope('relu1') as scope:
        relu1 = tf.nn.relu(tf.add(conv1, b1), name='relu1')

    # conv2+relu2
    with tf.name_scope('conv2') as scope:
        # w2 = tf.Variable(tf.initializers.TruncatedNormal(stddev=0.05)(shape=[3, 3, 32, 64]), name='w2')
        w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.05), name='w2')
        b2 = tf.Variable(tf.zeros(64), name='b2')
        conv2 = tf.nn.conv2d(relu1, w2, strides=[1, 2, 2, 1], padding='SAME', name='conv2')
    with tf.name_scope('relu2') as scope:
        relu2 = tf.nn.relu(conv2 + b2, name='relu2')

    # conv3+relu3
    with tf.name_scope('conv3') as scope:
        # w3 = tf.Variable(tf.initializers.TruncatedNormal(mean=0, stddev=0.05)(shape=[3, 3, 64, 128]), name='w3')
        w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.05), name='w3')
        b3 = tf.Variable(tf.zeros(128), name='b3')
        conv3 = tf.nn.conv2d(relu2, w3, strides=[1, 2, 2, 1], padding='SAME')
    with tf.name_scope('relu3') as scope:
        relu3 = tf.nn.relu(conv3 + b3, name='relu3')

    # full connect+relu
    with tf.name_scope('fc1') as scope:
        x_flat = tf.reshape(relu3, shape=[-1, 4 * 4 * 128])
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 128, 1024], stddev=0.05, mean=0), name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(1024), name='b_fc1')
        fc1 = tf.add(tf.matmul(x_flat, w_fc1), b_fc1)
    with tf.name_scope('relu_fc1') as scope:
        relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')

    # drop+f
    with tf.name_scope('drop_1') as scope:
        drop_1 = tf.nn.dropout(relu_fc1, keep_prob=keep_prob, name='drop_1')
    with tf.name_scope('bn_fc1') as scope:
        bn_fc1 = tf.layers.batch_normalization(drop_1, name='bn_fc1')
    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.05, mean=0), name='w_fc2')
        b_fc2 = tf.Variable(tf.zeros(512), name='b_fc2')
        fc2 = tf.add(tf.matmul(bn_fc1, w_fc2), b_fc2)
    with tf.name_scope('relu_fc2') as scope:
        relu_fc2 = tf.nn.relu(fc2, name='relu_fc2')
    with tf.name_scope('drop_2') as scope:
        drop_2 = tf.nn.dropout(relu_fc2, keep_prob=keep_prob, name='drop_2')
    with tf.name_scope('bn_fc2') as scope:
        bn_fc2 = tf.layers.batch_normalization(drop_2, name='bn_fc2')
    with tf.name_scope('fc3') as scope:
        w_fc3 = tf.Variable(tf.truncated_normal(shape=[512, 2], stddev=0.05, mean=0), name='w_fc3')
        b_fc3 = tf.Variable(tf.zeros(2), name='b_fc3')
        fc3 = tf.add(tf.matmul(bn_fc2, w_fc3), b_fc3)
    return fc3
