import shelve as sl
import random as rd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def datas():
    numdata = []
    numplace = []
    for i in range(1, 101):
        numgroup = np.zeros([10, 10], dtype=int)
        localx = rd.randint(0, 9)
        localy = rd.randint(0, 9)
        numgroup[localx][localy] = 1
        placenum = 10 * localx + localy
        numdata.append(numgroup)
        numplace.append([placenum])
    return numdata, numplace


def multip(x):
    # result1=tf.matmul(x,weigths_1)#+bias
    # out = tf.matmul(result1, weigths_2)
    out = tf.multiply(x, weigths_3)
    return out


x = tf.placeholder(tf.float64, [None, 10, 10], 'x')
y = tf.placeholder(tf.float64, [None, 1], 'y')
with tf.variable_scope('test1', reuse=tf.AUTO_REUSE):
    weigths_1 = tf.get_variable('w1', [10, 1], tf.float64)
    weigths_2 = tf.get_variable('w2', [1, 10], tf.float64)
    weigths_3 = tf.get_variable('w3', [10, 10], dtype=tf.float64)
    weigths_4=tf.matmul(weigths_1,weigths_2)
    # bias=tf.get_variable('b',[10,1],tf.float64)
out = tf.map_fn(multip, x)
out1 = tf.reduce_sum(out, 2)
out2 = tf.reduce_sum(out1, 1)
out2=tf.reshape(out2,[-1,1])
# weigths_2=tf.get_variab
# le('w2',[10,1],tf.float64)
# w1=layers.fully_connected(x,1)
# w2=layers.fully_connected(w1,10)
# w5=layers.fully_connected(w2,1)+bias
# y_hat=tf.reduce_mean(weigths_1,1)
loss = tf.sqrt(tf.reduce_sum(tf.square(out2 - y)))
# loss=tf.sqrt(tf.square(tf.subtract(y_hat,y)))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()
s = [100, 1]

with tf.Session() as sess:
    sess.run(init)
    while (s[0] > 1):
        x_batch, y_batch = datas()
        s = sess.run([loss, train], feed_dict={x: x_batch, y: y_batch})
        print(s[0])
        # print("s",sess.run([y,out2],feed_dict={x:x_batch,y:y_batch}))
    variable_names = tf.trainable_variables()
    values = sess.run('test1/w3:0')
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)
    print(values)
