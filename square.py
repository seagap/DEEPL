import cv2 as cv
import numpy as np
import random as rd
import math
import tensorflow as tf
import tensorflow.contrib.layers as layers
def randata():
    data=[]
    info=[]
    for i in range (1,11):
        img = cv.imread('ssd.jpg')
        imgw = len(img[0])
        imgh = len(img[1])
        border = min(imgw, imgh)
        sidelen = rd.randint(3, border)
        px = rd.randint(3, imgw - sidelen)
        py = rd.randint(3, imgh - sidelen)
        imgdata = img[py:py + sidelen, px:px + sidelen]
        deltax = imgw / 2 - px - sidelen / 2
        deltay = imgh / 2 - py - sidelen / 2
        identy = math.sqrt(
            deltax ** 2 + deltay ** 2
        )
        ratio = (sidelen ** 2) / (imgw * imgh)
        deltax /= identy
        deltay /= identy
        imginfo = [deltax, deltay, ratio]
        imgdata=cv.resize(imgdata,(16,16))
        data.append(imgdata)
        info.append(imginfo)
    return data, info
s,r=randata()
print(np.shape(s),np.shape(r))

x=tf.placeholder(tf.float64,[None,16,16,3],'x')
y=tf.placeholder(tf.float64,[None,3],'y')
l1=layers.conv2d(x,100,[16,16],16,padding='VALID')
l2=layers.conv2d(l1,3,[1,1],16,padding='VALID')
l3=tf.reduce_sum(l2,1)
l4=tf.reduce_sum(l3,1)

loss=tf.reduce_sum(tf.sqrt(tf.square(l4-y)))
train=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init=tf.global_variables_initializer()
s=[100]
with tf.Session() as sess:
    sess.run(init)
    while(s[0]>10):
        xbatch,ybatch=randata()
        s =sess.run([loss,train],feed_dict={x:xbatch,y:ybatch})
        print(s[0])