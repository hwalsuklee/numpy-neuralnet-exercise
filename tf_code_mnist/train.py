from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models
import mnist_data
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs/train"

# user input
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of various cnn models"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model', type=str, default='GAN',
                        choices=['baseline', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
                        help='The type of model', required=True)

    return parser.parse_args()

# main function
def main():
    # parse arguments
    args = parse_args()

    # Some parameters
    data_augmentation = False
    batch_size = 10

    training_epochs = 25
    num_labels = mnist_data.NUM_LABELS

    # Choose model
    if args.model == 'baseline':
        model = models.baseline
        learning_rate = 0.1
        display_step = 500
    elif args.model == 'v1':
        model = models.version_1
        learning_rate = 0.1
        display_step = 500
    elif args.model == 'v2':
        model = models.version_2
        learning_rate = 0.1
        display_step = 500
    elif args.model == 'v3':
        model = models.version_3
        learning_rate = 0.03
        display_step = 500
    elif args.model == 'v4':
        model = models.version_3
        learning_rate = 0.03
        data_augmentation = True
        display_step = 2500
    elif args.model == 'v5':
        model = models.version_5
        learning_rate = 0.03
        data_augmentation = True
        display_step = 2500
    elif args.model == 'v6':
        model = models.version_6
        learning_rate = 0.03
        data_augmentation = True
        display_step = 5500
    else:
        NotImplementedError()

    # Prepare mnist data
    train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(
        data_augmentation)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10]) #answer

    # Predict
    pred, pred_logit = model(x, is_training)

    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred_logit))

    # Define optimizer
    with tf.name_scope("ADAM"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(train_size / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Loop for epoch
    for epoch in range(training_epochs):

        # Random shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-num_labels]
        train_labels_ = train_total_data[:, -num_labels:]

        # Loop over all batches
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , feed_dict={x: batch_xs, y: batch_ys, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%02d,' % (epoch + 1),
                      "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

        # Get accuracy for test data
        print("Test accuracy at Epoch %02d : %g" % (epoch+1, accuracy.eval(
            feed_dict={x: test_data, y: test_labels, is_training: False})))

    # Calculate accuracy for all mnist test images
    print("Test accuracy for the latest result: %g" % accuracy.eval(
        feed_dict={x: test_data, y: test_labels, is_training: False}))

if __name__ == '__main__':
    main()