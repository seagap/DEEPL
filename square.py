import cv2 as cv
import numpy as np
import random as rd
import math
import tensorflow as tf
import tensorflow.contrib.layers as layers


def randata():
    data = []
    info = []
    # adjust batch size*****************************************
    for i in range(1, 51):
        img = cv.imread('23.jpg')
        imgw = len(img[0])
        imgh = len(img)
        border = min(imgw, imgh)
        sidelen = 50  # rd.randint(6, border)
        # px = rd.randint(0, imgw - sidelen)
        # py = rd.randint(0, imgh - sidelen)
        px = 80
        py = 80
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


# net argument *****************************************
x = tf.placeholder(tf.float64, [None, 16, 16, 3], 'x')
y = tf.placeholder(tf.float64, [None, 1], 'y')
b = tf.get_variable('b', [2], dtype=tf.float64)

l1 = layers.conv2d(x, 256, [3,3], (3,3), padding='VALID')
# l2 = layers.conv2d(l1, 16, [1, 1], (1, 1), padding='VALID',activation_fn=tf.nn.softmax)
l2_1 = layers.fully_connected(l1, 16)
l2_2 = layers.fully_connected(l2_1, 16)
l2_3 = layers.fully_connected(l2_2,1)
l3 = tf.reduce_sum(l2_3, 1)
l4 = tf.reduce_sum(l3, 1)
l5 = l4
loss = tf.reduce_sum(tf.sqrt(tf.square(l5 - y)))
# train = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
train = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.98).minimize(loss)
init = tf.global_variables_initializer()

s = [100]
with tf.Session() as sess:
    sess.run(init)
    var = tf.trainable_variables()
    print(var)
    while (s[0] >1):
        xbatch, ybatch = randata()
        s = sess.run([loss, train], feed_dict={x: xbatch, y: ybatch})
        # t = sess.run(l4, feed_dict={x: xbatch, y: ybatch})
        # print(np.shape(t))
        print(s[0])
    xbatch, ybatch = randata()
    lasty = l4.eval({x: xbatch, y: ybatch})
    print('real', ybatch)
    print('predict', lasty)
    for i in range(0, 10):
        print(i)
        imgtemp = cv.resize(xbatch[i], (320, 320))
        cv.imshow('show', imgtemp)
        cv.waitKey(0)
