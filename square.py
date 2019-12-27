import cv2 as cv
import numpy as np
import random as rd
import math
import tensorflow as tf
import tensorflow.contrib.layers as layers
from random import choice


def randata():
    data = []
    info = []
    # adjust batch size*****************************************
    for i in range(1, 201):
        img = cv.imread('23.jpg')
        imgw = len(img[0])
        imgh = len(img)
        border = min(imgw, imgh)
        sidelen = 50  # rd.randint(6, border)
        # px = rd.randint(0, imgw - sidelen)
        # py = rd.randint(0, imgh - sidelen)
        #region multiple is sample num
        # px = rd.randint(80, 96)
        # py = rd.randint(80, 96)
        px = choice([86,32,58,200,36,15,28,65,79,85,23,14,64,62,89,12])
        py = choice([86,32,58,200,36,15,28,65,79,85,23,14,64,62,89,12])
        imgdata = img[py:py + sidelen, px:px + sidelen]
        deltax = imgw / 2 - px - sidelen / 2
        deltay = imgh / 2 - py - sidelen / 2
        try:
            tan = math.atan2(deltay,(deltax+0.00001))/math.pi
        except ZeroDivisionError:
            print(imgw,imgh,px,py)
        # identy = math.sqrt(
        #     deltax ** 2 + deltay ** 2
        # )+0.00001
        # ratio = (sidelen ** 2) / (imgw * imgh)
        # deltax /= identy
        # deltay /= identy
        imginfo = np.array([tan])
        try:
            imgdata = cv.resize(imgdata, (16, 16))
        except cv.error:
            print("error:*******************", px, py, np.shape(imgdata), sidelen, imginfo)
            print("error:hw", imgw, imgh)
        data.append(imgdata)
        info.append(imginfo)
    info=np.array(info)
    return data, info

graph = tf.Graph()
with graph.as_default():
    # net argument *****************************************
    x = tf.placeholder(tf.float64, [None, 16, 16, 3], 'x')
    y = tf.placeholder(tf.float64, [None, 1], 'y')
    b = tf.get_variable('b', [2], dtype=tf.float64)

    l1 = layers.conv2d(x, 256, [3,3], (3,3), padding='VALID')
    # l2 = layers.conv2d(l1, 16, [1, 1], (1, 1), padding='VALID',activation_fn=tf.nn.softmax)
    l2_1 = layers.fully_connected(l1, 16,activation_fn=tf.nn.relu)
    l2_2 = layers.fully_connected(l2_1, 16)
    l2_3 = layers.fully_connected(l2_2,1,activation_fn=tf.nn.sigmoid)
    l3 = tf.reduce_sum(l2_3, 1)
    l4 = tf.reduce_sum(l3, 1)
    l5 = l4
    loss = tf.reduce_sum(tf.sqrt(tf.square(l5 - y)))
    train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
    # train = tf.train.MomentumOptimizer(learning_rate=0.000001,momentum=0.98).minimize(loss)
    init = tf.global_variables_initializer()

    s = [100]
    with tf.Session() as sess:
        sess.run(init)
        var = tf.trainable_variables()
        print(var)
        while (s[0] >5):
            xbatch, ybatch = randata()
            s = sess.run([loss,l5, train], feed_dict={x: xbatch, y: ybatch})
            # t = sess.run(l4, feed_dict={x: xbatch, y: ybatch})
            # print(np.shape(t))
            print("loss",s[0])
        print("realy", ybatch,"predict",s[1])
        # xbatch, ybatch = randata()
        # lastl2 = l2_3.eval({x: xbatch, y: ybatch})
        # lastl3 = l3.eval({x: xbatch, y: ybatch})
        # lastl4 = l4.eval({x: xbatch, y: ybatch})
        # print('real', ybatch)
        # # print('predict \n l2', lastl2,"\n l3",lastl3,"\n l4",lastl4)
        # print('predic', lastl4)
        # for i in range(0, 10):
        #     print(i)
        #     imgtemp = cv.resize(xbatch[i], (320, 320))
        #     cv.imshow('show', imgtemp)
        #     cv.waitKey(0)
    # filewrite=tf.summary.FileWriter("./pic",graph)
