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
        img = cv.imread('ssd.jpg')
        imgw = len(img[0])
        imgh = len(img)
        border = min(imgw, imgh)
        sidelen = 50  # rd.randint(6, border)
        px = rd.randint(0, imgw - sidelen)
        py = rd.randint(0, imgh - sidelen)
        imgdata = img[py:py + sidelen, px:px + sidelen]
        deltax = imgw / 2 - px - sidelen / 2
        deltay = imgh / 2 - px - sidelen / 2
        identy = math.sqrt(
            deltax ** 2 + deltay ** 2
        )
        ratio = (sidelen ** 2) / (imgw * imgh)
        deltax /= identy
        deltay /= identy
        imginfo = np.array([deltax, deltay])
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
y = tf.placeholder(tf.float64, [None, 2], 'y')
b = tf.get_variable('b', [2], dtype=tf.float64)

l1 = layers.conv2d(x, 256, [16, 16], (16, 16), padding='VALID')
l1_1 = layers.fully_connected(l1, 16)
l2 = layers.conv2d(l1_1, 2, [1, 1], (1, 1), padding='VALID', activation_fn=tf.nn.softmax)
l3 = tf.reduce_sum(l2, 1)
l4 = tf.reduce_sum(l3, 1)
l5 = l4
loss = tf.reduce_sum(tf.sqrt(tf.square(l5 - y)))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
s = [100]
with tf.Session() as sess:
    sess.run(init)
    var = tf.trainable_variables()
    print(var)
    while (s[0] >20):
        xbatch, ybatch = randata()
        s = sess.run([loss, train], feed_dict={x: xbatch, y: ybatch})
        # s = sess.run(l1, feed_dict={x: xbatch, y: ybatch})
        print(s[0])
    xbatch, ybatch = randata()
    lasty = l5.eval({x: xbatch, y: ybatch})
    print('real', ybatch)
    print('predict', lasty)
    for i in range(0, 10):
        print(i)
        imgtemp = cv.resize(xbatch[i], (320, 320))
        cv.imshow('show', imgtemp)
        cv.waitKey(0)
