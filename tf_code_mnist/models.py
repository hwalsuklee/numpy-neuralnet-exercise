from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

def version_0(inputs, is_training):
    with tf.name_scope('version_0'):
        n_hidden = 100
        n_out = 10

        # 100 sigmoid neurons
        net = slim.fully_connected(inputs, n_hidden, scope='fc1',
                                   activation_fn=tf.nn.sigmoid,
                                   weights_initializer=initializers.xavier_initializer(),
                                   biases_initializer=init_ops.zeros_initializer())

        # 10 neurons (softmax)
        logits = slim.fully_connected(net, n_out, activation_fn=None, scope='fco',
                                      weights_initializer=initializers.xavier_initializer(),
                                      biases_initializer=init_ops.zeros_initializer())
        out_layer = tf.nn.softmax(logits)

    return out_layer, logits

def version_1(inputs, is_training):
    with tf.name_scope('version_1'):
        n_filter = 20
        n_hidden = 100
        n_out = 10

        # Reshaping for convolutional operation
        x = tf.reshape(inputs, [-1, 28, 28, 1])

        # Convolutional layer
        net = slim.conv2d(x, n_filter, [5, 5], padding='VALID', activation_fn = tf.nn.sigmoid, scope='conv1')

        # Pooling layer
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')

        # Flatten for fully-connected layer
        net = slim.flatten(net, scope='flatten3')

        # 100 sigmoid neurons
        net = slim.fully_connected(net, n_hidden, scope='fc1',
                                   activation_fn=tf.nn.sigmoid,
                                   weights_initializer=initializers.xavier_initializer(),
                                   biases_initializer=init_ops.zeros_initializer())

        # 10 neurons (softmax)
        logits = slim.fully_connected(net, n_out, activation_fn=None, scope='fco',
                                      weights_initializer=initializers.xavier_initializer(),
                                      biases_initializer=init_ops.zeros_initializer())
        out_layer = tf.nn.softmax(logits)

    return out_layer, logits

def version_2(inputs, is_training):
    with tf.name_scope('version_2'):
        n_filter1 = 20
        n_filter2 = 40
        n_hidden = 100
        n_out = 10

        # Reshaping for convolutional operation
        x = tf.reshape(inputs, [-1, 28, 28, 1])

        # Convolutional layer
        net = slim.conv2d(x, n_filter1, [5, 5], padding='VALID', activation_fn = tf.nn.sigmoid, scope='conv1')

        # Pooling layer
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')

        # Convolutional layer
        net = slim.conv2d(net, n_filter2, [5, 5], padding='VALID', activation_fn=tf.nn.sigmoid, scope='conv2')

        # Pooling layer
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')

        # Flatten for fully-connected layer
        net = slim.flatten(net, scope='flatten3')

        # 100 sigmoid neurons
        net = slim.fully_connected(net, n_hidden, scope='fc1',
                                   activation_fn=tf.nn.sigmoid,
                                   weights_initializer=initializers.xavier_initializer(),
                                   biases_initializer=init_ops.zeros_initializer())

        # 10 neurons (softmax)
        logits = slim.fully_connected(net, n_out, activation_fn=None, scope='fco',
                                      weights_initializer=initializers.xavier_initializer(),
                                      biases_initializer=init_ops.zeros_initializer())
        out_layer = tf.nn.softmax(logits)

    return out_layer, logits

def version_3(inputs, is_training):
    with tf.name_scope('version_3'):
        n_filter1 = 20
        n_filter2 = 40
        n_hidden = 100
        n_out = 10

        # L2 Regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.1)):

            # Reshaping for convolutional operation
            x = tf.reshape(inputs, [-1, 28, 28, 1])

            # Convolutional layer
            net = slim.conv2d(x, n_filter1, [5, 5], padding='VALID', activation_fn = tf.nn.relu, scope='conv1')

            # Pooling layer
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')

            # Convolutional layer
            net = slim.conv2d(net, n_filter2, [5, 5], padding='VALID', activation_fn=tf.nn.relu, scope='conv2')

            # Pooling layer
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')

            # Flatten for fully-connected layer
            net = slim.flatten(net, scope='flatten3')

            # 100 ReLu neurons
            net = slim.fully_connected(net, n_hidden, scope='fc1',
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=initializers.xavier_initializer(),
                                       biases_initializer=init_ops.zeros_initializer())

            # 10 neurons (softmax)
            logits = slim.fully_connected(net, n_out, activation_fn=None, scope='fco',
                                          weights_initializer=initializers.xavier_initializer(),
                                          biases_initializer=init_ops.zeros_initializer())
            out_layer = tf.nn.softmax(logits)

    return out_layer, logits

def version_5(inputs, is_training):
    with tf.name_scope('version_5'):
        n_filter1 = 20
        n_filter2 = 40
        n_hidden1 = 100
        n_hidden2 = 100
        n_out = 10

        # L2 Regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.1)):
            # Reshaping for convolutional operation
            x = tf.reshape(inputs, [-1, 28, 28, 1])

            # Convolutional layer
            net = slim.conv2d(x, n_filter1, [5, 5], padding='VALID', activation_fn = tf.nn.relu, scope='conv1')

            # Pooling layer
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')

            # Convolutional layer
            net = slim.conv2d(net, n_filter2, [5, 5], padding='VALID', activation_fn=tf.nn.relu, scope='conv2')

            # Pooling layer
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')

            # Flatten for fully-connected layer
            net = slim.flatten(net, scope='flatten3')

            # 100 ReLu neurons
            net = slim.fully_connected(net, n_hidden1, scope='fc1',
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=initializers.xavier_initializer(),
                                       biases_initializer=init_ops.zeros_initializer())

            # 100 ReLu neurons
            net = slim.fully_connected(net, n_hidden2, scope='fc2',
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=initializers.xavier_initializer(),
                                       biases_initializer=init_ops.zeros_initializer())

            # 10 neurons (softmax)
            logits = slim.fully_connected(net, n_out, activation_fn=None, scope='fco',
                                          weights_initializer=initializers.xavier_initializer(),
                                          biases_initializer=init_ops.zeros_initializer())
            out_layer = tf.nn.softmax(logits)

    return out_layer, logits

def version_6(inputs, is_training):
    with tf.name_scope('version_6'):
        n_filter1 = 20
        n_filter2 = 40
        n_hidden1 = 100
        n_hidden2 = 100
        n_out = 10

        # # L2 Regularizer
        # with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                     weights_regularizer=slim.l2_regularizer(0.1)):
        # batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        # with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                     normalizer_fn=slim.batch_norm,
        #                     normalizer_params=batch_norm_params,
        #                     weights_regularizer=slim.l2_regularizer(0.1)):

        # Reshaping for convolutional operation
        x = tf.reshape(inputs, [-1, 28, 28, 1])

        # Convolutional layer
        net = slim.conv2d(x, n_filter1, [5, 5], padding='VALID', activation_fn = tf.nn.relu, scope='conv1')

        # Pooling layer
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')

        # Convolutional layer
        net = slim.conv2d(net, n_filter2, [5, 5], padding='VALID', activation_fn=tf.nn.relu, scope='conv2')

        # Pooling layer
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')

        # Flatten for fully-connected layer
        net = slim.flatten(net, scope='flatten3')

        # 100 ReLu neurons
        net = slim.fully_connected(net, n_hidden1, scope='fc1',
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=initializers.xavier_initializer(),
                                   biases_initializer=init_ops.zeros_initializer())
        net = slim.dropout(net, is_training=is_training, scope='dropout1')  # 0.5 by default

        # 100 ReLu neurons
        net = slim.fully_connected(net, n_hidden2, scope='fc2',
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=initializers.xavier_initializer(),
                                   biases_initializer=init_ops.zeros_initializer())
        net = slim.dropout(net, is_training=is_training, scope='dropout2')  # 0.5 by default

        # 10 neurons (softmax)
        net = slim.fully_connected(net, n_out, activation_fn=None, scope='fco',
                                      weights_initializer=initializers.xavier_initializer(),
                                      biases_initializer=init_ops.zeros_initializer())
        logits = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        out_layer = tf.nn.softmax(logits)

    return out_layer, logits