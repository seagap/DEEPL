#  -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.initializers
import shelve
import numpy as np

# defined network
ly1_hidden = np.zeros(shape=(30, 50), dtype=None)
ly2_hidden = np.zeros(shape=(30, 50), dtype=None)
out = 2
eta = 0.001
epoch = 0
imgcenter = shelve.open('imginfo')


# load data
def generdata():
    global epoch
    imgdata = []
    ydata = []
    for i in range(epoch * 10 + 1, (epoch + 1) * 10):  # epoch * 10+1, (epoch + 1) * 10
        img = cv2.imread('../generousimg/' + str(i) + '.jpg')
        img = cv2.resize(img, (500, 300))
        imgdata.append(img)
        ydata.append(imgcenter[str(i)]['cenx'])
    epoch += 1
    return imgdata, ydata


#
# aa,bb=generdata()
# cv2.imshow('img',aa[0])
# cv2.waitKey(0)
# img = cv2.imread('s.jpg')
# img = cv2.resize(img, (500, 300))
# print(np.shape(img))


# cv2.imshow('windows',img)
# cv2.waitKey()
# print(np.shape(img))

def mutilayers(x):
    kernel = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    ly1 = layers.conv2d(x, kernel_size=tuple([3, 3]), stride=([3, 3]), padding='SAME',
                        weights_initializer=tf.initializers.constant([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                                                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                                      [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
                        num_outputs=1, trainable=False, )
    ly2 = layers.conv2d(x, kernel_size=tuple([3, 3]), stride=([3, 3]), padding='SAME',
                        weights_initializer=tf.initializers.constant([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                                      [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]),
                        num_outputs=1, trainable=False, )
    ly3 = layers.conv2d(x, kernel_size=tuple([3, 3]), stride=([3, 3]), padding='SAME',
                        weights_initializer=tf.initializers.constant([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                                                      [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                                                      [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]),
                        num_outputs=1, trainable=False, )
    ly4 = layers.conv2d(x, kernel_size=tuple([3, 3]), stride=([3, 3]), padding='SAME',
                        weights_initializer=tf.initializers.constant([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                                                                      [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                                                                      [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]),
                        num_outputs=1, trainable=False, )
    ly5 = ly1 + ly2 + ly3 + ly4
    ly6 = layers.max_pool2d(ly5, kernel_size=([3, 3]))
    ly7 = layers.conv2d(ly6, kernel_size=tuple([3, 3]), stride=([1, 1]), padding='SAME',
                        num_outputs=1)
    ly8 = layers.max_pool2d(ly7, kernel_size=([3, 3]))
    ly9 = layers.conv2d(ly8, kernel_size=tuple([24, 41]), stride=([1, 1]), padding='valid',
                        num_outputs=1)
    return ly9


#
# #-------------------------------------------------------
# # x = tf.placeholder(dtype=tf.float64, shape=[None, 300, 500, 3], name='x')
# # y = mutilayers(x / 255.0)
# # init = tf.global_variables_initializer()
# # with tf.Session() as sess:
# #     sess.run(init)
# #     tempshape=sess.run(y,feed_dict={x:[img]})
# # print(np.shape(tempshape))
# # ----------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float64, shape=[None, 300, 500, 3], name='x')
    y = tf.placeholder(dtype=tf.float64, shape=[None], name='y')
    # img.dtype='float'
    # print(img)

    y_hat = mutilayers(x / 255.0)
    loss = y_hat[0][0][0][0] - y
    train = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, 10):
            x_batch, y_batch = generdata()
            print(np.shape(x_batch))
            print(np.shape(y_batch))
            sd = sess.run([loss, train], feed_dict={x: x_batch, y: y_batch})
correct_prediction = tf.equal(y_hat, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
xtest_batch, ytest_batch = generdata()
print("Accuracy：", accuracy.eval({x: xtest_batch, y: ytest_batch}))
# file_writer = tf.summary.FileWriter('./log3', graph)
