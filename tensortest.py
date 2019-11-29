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

# load data
img = cv2.imread('s.jpg')
img = cv2.resize(img, (500, 300))
print(np.shape(img))


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

# x = tf.placeholder(dtype=tf.float64, shape=[None, 300, 500, 3], name='x')
# y = mutilayers(x / 255.0)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     tempshape=sess.run(y,feed_dict={x:[img]})
# print(np.shape(tempshape))
#----------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float64, shape=[None, 300, 500, 3], name='x')
    # img.dtype='float'
    # print(img)

    y = mutilayers(x / 255.0)
    loss = y[0][0][0][0] - 0.5
    train = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sd = sess.run([loss, train], feed_dict={x: [img]})

file_writer = tf.summary.FileWriter('./log3', graph)
# print(tf.get_default_graph())
# print(np.shape(sd))
# print(np.amax(sd))
# # ss = sd[0] * 255 / np.amax(sd)
# # ss = ss.astype(np.int16)
# # print(np.shape(ss))
# # cv2.imshow('windows', ss)
# # cv2.waitKey()
# # cv2.imwrite('ss.jpg', ss)

#
# cv2.imshow('test', mutilayers(img))
# imginfo=shelve.open('imginfo')
