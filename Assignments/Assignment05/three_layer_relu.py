# Three Layered fnn relu
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import scipy
import time
from util import func_confusion_matrix
from model_configs import evaluation, relu_three_layer_inference, loss, training, report
from data_helpers import gen_batch

# load (downloaded if needed) the MNIST dataset # Downloaded Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
CLASSES = 10  # digit labels can be a value from 0 - 9

# transform each image from 28 by 28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2] #think multiply as 2d row by col operation?
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')  # .astype('float32') for the normalization
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')
# normalize inputs from gray scale of 0-255 to values between 0-1

x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 5
# For every image, we use its pixel-wise intensities as features

# splitting of data
total_num_samples = len(x_train)
random_idx_order = np.random.permutation(total_num_samples)
train_x = x_train[random_idx_order[0:50000], :]
train_y = y_train[random_idx_order[0:50000]]
validation_x = x_train[random_idx_order[50000:], :]
validation_y = y_train[random_idx_order[50000:]]

# dictionary map of sets to work with
data = {
    'images_train': train_x,
    'labels_train': train_y,
    'images_validation': validation_x,
    'labels_validation': validation_y,
    'images_test': x_test,
    'labels_test': y_test
  }

# Hyper-parameters
flags = tf.flags  # declare flags to be the tf.flags module ( interface that utilizes abseil-py; absl.flags)
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 800, 'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_float('reg_constant', 0.01, 'Regularization constant.')
FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters")
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr, value))

start_time = time.time()

""" 
placeholder declaration: tf.placeholder(dtype, shape=None, name=None)
Note: Passing None to a shape argument of a tf.placeholder tells it simply that that dimension is unspecified, 
    and to infer that dimension from the tensor you are feeding it during run-time. Only some arguments 
    (generally the batch_size argument) can be set to None since Tensorflow needs to be able to construct a working
     graph before run-time. This is useful for when you don't want to specify a batch_size before run time.
"""
images_placeholder = tf.placeholder(tf.float32, shape=[None, pixel_count], name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')
logits = relu_three_layer_inference(images_placeholder, pixel_count, FLAGS.hidden1, FLAGS.hidden2, CLASSES, reg_constant=FLAGS.reg_constant) # build model
loss = loss(logits, labels_placeholder)
train_step = training(loss, FLAGS.learning_rate)
accuracy = evaluation(logits, labels_placeholder)
report = report(logits, labels_placeholder)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    zipped_data = zip(data['images_train'], data['labels_train'])
    batches = gen_batch(list(zipped_data), FLAGS.batch_size, FLAGS.max_steps)

    for i in range(FLAGS.max_steps):
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)
        feed_dict = {
            images_placeholder: images_batch,
            labels_placeholder: labels_batch
        }
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))

        sess.run([train_step, loss], feed_dict=feed_dict)

    # validation set evaluation
    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data['images_test'],
        labels_placeholder: data['labels_test']})  #'numpy.float32' object is not iterable
    print('Test accuracy {:g}'.format(test_accuracy))
    true_false_prediction, prediction_matrix = sess.run(report, feed_dict={
        images_placeholder: data['images_test'],
        labels_placeholder: data['labels_test']})

    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(prediction_matrix, y_test) # func modified due to tensors
    print("confusion matrix\n", conf_matrix)
    print("recall_array\n", recall_array)
    print("precision_array\n", precision_array)

    # Ten Images Model Made Errors
    error_count = 1  # start at 1 not 0
    falsely_predicted_indexes = [i for i, x in enumerate(true_false_prediction) if x - 1]  # if x = 1 (correct); 1 - 1 == false
    fig = plt.figure(None, (10, 10))
    for index in falsely_predicted_indexes:
        # convert vector.shape(748) back to (28,28) so we can see data
        image = np.reshape(x_test[index], (28, 28))
        fig.add_subplot(5, 2, error_count) # rows(first arg); cols(second arg); sub_plt_number(third arg)
        plt.imshow(image)
        error_count += 1
        if error_count == 11:
            break
    plt.show()

    total_time = time.time() - start_time
    print("Total time: {}".format(total_time))

