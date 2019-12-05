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
    ly7 = layers.fully_connected(ly6,activation_fn=tf.nn.relu,num_outputs=1)
    # ly8=tf.reshape(ly7,[9,-1,16])
    # right layer
    ly8 = tf.reshape(ly6, [9, -1, 1])
    ly9 = layers.fully_connected(ly8,activation_fn=tf.nn.relu, num_outputs=1)
    test=tf.reshape(ly9,[9,-1])
    ly10 = layers.fully_connected(test,activation_fn=tf.nn.relu, num_outputs=3)
    # leftout=layers.dense_to_sparse()
    #kernel
    # ly11 = layers.fully_connected(ly6,activation_fn=tf.nn.relu, num_outputs=256)
    # ly12 = layers.fully_connected(ly11, activation_fn=tf.nn.relu, num_outputs=256)
    # ly13 = tf.reshape(ly12, [9, 16, -1])
    # outmatrix1=tf.matmul(ly8,ly13)
    # outmatrix2 = tf.matmul(outmatrix1, ly10)
    # outputs=layers.conv2d(ly6,kernel_size=tuple([49,83]), stride=([1,1]), padding='SAME',
    #                     weights_initializer=tf.initializers.constant(outmatrix2),num_outputs=1)
    # ly10=ly9[0][0][0][0]   #Dose it have batch var ?  should not have
    return ly10


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
    loss = y_hat - y
    # train = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, 2):
            x_batch, y_batch = generdata()
            # print(np.shape(x_batch))
            # print(np.shape(y_batch))
            # sd = sess.run([loss, train], feed_dict={x: x_batch, y: y_batch})
            print(np.shape(sess.run(y_hat, feed_dict={x: x_batch})))
# correct_prediction = tf.equal(y_hat, y)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# xtest_batch, ytest_batch = generdata()
# print("Accuracy：", accuracy.eval({x: xtest_batch, y: ytest_batch}))
# file_writer = tf.summary.FileWriter('./log3', graph)
