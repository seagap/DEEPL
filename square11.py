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
        s=rd.randint(0,14909)
        img = cv.imread('../validation/posdata/'+str(s)+'.jpg')
        imgw = len(img[0])
        imgh = len(img)
        border = min(imgw, imgh)
        sidelen = rd.randint(5,border) # rd.randint(6, border)
        px = rd.randint(0, imgw - sidelen)
        py = rd.randint(0, imgh - sidelen)
        #region multiple is sample num
        # px = rd.randint(81, 83)
        # py = rd.randint(81, 83)
        # px = choice([86,32,58,200,36,15,28,65,79,85,23,14,64,62,89,12])
        # py = choice([86,32,58,200,36,15,28,65,79,85,23,14,64,62,89,12])
        # px = choice([86,32])
        # py = choice([89,12])
        imgdata = img[py:py + sidelen, px:px + sidelen]
        deltax = px + sidelen / 2
        deltay = py + sidelen / 2
        try:
            tan1 = math.atan2(deltay,(deltax+0.00001))/math.pi
            tan2 = math.atan2((deltay-imgh), (deltax-imgw + 0.00001)) / math.pi
        except ZeroDivisionError:
            print(imgw,imgh,px,py)
        # identy = math.sqrt(
        #     deltax ** 2 + deltay ** 2
        # )+0.00001
        # ratio = (sidelen ** 2) / (imgw * imgh)
        # deltax /= identy
        # deltay /= identy
        imginfo = np.array([tan1,tan2])
        try:
            imgdata = cv.resize(imgdata, (16, 16))
        except cv.error:
            print("error:*******************", px, py, np.shape(imgdata), sidelen, imginfo)
            print("error:hw", imgw, imgh)
        data.append(imgdata)
        info.append(imginfo)
    info=np.array(info)
    return data, info

def resort(x):
    batch_size = x.get_shape().as_list()[0]
    x = tf.transpose(x, perm=[2,1,0])
    x = tf.nn.top_k(x, k=batch_size ).values
    x = tf.transpose(x, perm=[2, 1, 0])
    batch_size = x.get_shape().as_list()[2]
    x = tf.nn.top_k(x, k=batch_size).values
    return x
# net argument *****************************************
x = tf.placeholder(tf.float64, [None, 16, 16, 3], 'x')
y = tf.placeholder(tf.float64, [None, 2], 'y')
b = tf.get_variable('b', [2], dtype=tf.float64)

l1 = layers.conv2d(x, 16, [4,4], (1,1), padding='VALID')
l1_0 = layers.conv2d(x, 256, [4,4], (3,3), padding='VALID')
l2_0=tf.map_fn(resort,l1_0)
l2=layers.conv2d(l2_0,16,[5,5],(1,1), padding='VALID')
l3 = layers.fully_connected(l2,256,activation_fn=tf.nn.sigmoid)
l3_0 = layers.fully_connected(l3,64 )
l3_1 = layers.fully_connected(l3_0,2,activation_fn=tf.nn.sigmoid)*2-1
l3_2 = tf.reduce_sum(l3_1, 1)
l4 = tf.reduce_sum(l3_2, 1,name="yhat")
# l5 = l4
loss = tf.reduce_sum(tf.sqrt(tf.square(l4 - y)))
train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
init = tf.global_variables_initializer()
s = [100]
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,save_path="./models/model")
    sess.run(init)
    var = tf.trainable_variables()
    print(var)
    while (s[0] >1):
        xbatch, ybatch = randata()
        s = sess.run([loss,l4, train], feed_dict={x: xbatch, y: ybatch})
        # s = sess.run(l4, feed_dict={x: xbatch, y: ybatch})
        print(s[0])
    print("realy", ybatch, "predict", s[1])
    saver.save(sess,save_path="./model/model")
    # xbatch, ybatch = randata()
    # lasty = t4.eval({x: xbatch, y: ybatch})
    # print('real', ybatch)
    # print('predict', lasty)
    # for i in range(0, 10):
    #     print(i)
    #     imgtemp = cv.resize(xbatch[i], (320, 320))
    #     cv.imshow('show', imgtemp)
    #     cv.waitKey(0)
