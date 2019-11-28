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

# load data
img = cv2.imread('s.jpg')
img = cv2.resize(img, (500, 300))
print(np.shape(img))

# cv2.imshow('windows',img)
# cv2.waitKey()
# print(np.shape(img))
def mutilayers(x):
    kernel = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # ly1 = layers.conv2d(x, kernel_size=tuple([3, 3]),
    #                     weights_initializer=tf.initializers.constant([[1,1,1],[1,1,1],[1,1,1]]),
    #                     num_outputs=1,trainable=False)
    ly1 = layers.conv2d(x, kernel_size=tuple([2,2]),stride=([1,1]),
                        weights_initializer=tf.initializers.constant([[[1,1,1],
                                                                      [1,1,1],
                                                                      [1,1,1]]]),
                        num_outputs=3,trainable=False,)
    return ly1*255/12


x = tf.placeholder(dtype=tf.float64, shape=[None, 300,500, 3], name='x')

# img.dtype='float'
# print(img)
y = mutilayers(x/255.0)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sd=sess.run(y, feed_dict={x:[img]})
ss=sd[0].astype(np.int16)
print(ss)
print(np.shape(ss))
cv2.imshow('windows',ss)
cv2.waitKey()
cv2.imwrite('ss.jpg', ss)

#
# cv2.imshow('test', mutilayers(img))
# imginfo=shelve.open('imginfo')
