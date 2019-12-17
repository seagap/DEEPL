import tensorflow as tf
import numpy as np

m1 = tf.get_variable('m1', [10, 3], tf.float64, initializer=tf.initializers.constant([
    [1, 0.1, 0.1],
    [0.1, 1, 0.1],
    [0.1, 0.1, 1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
]))

m2 = tf.constant([[1, 5, 9], [2, 3, 7], [6, 5, 7]], dtype=tf.float64)
m3 = tf.get_variable('m3', [3, 10], tf.float64, initializer=tf.initializers.constant([
    [1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
], dtype=tf.float64))
y = tf.constant([[1, 5, 9, 0, 0, 0, 0, 0, 0, 0],
                 [2, 3, 7, 0, 0, 0, 0, 0, 0, 0],
                 [6, 5, 7, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float64
                )

y_hat = tf.matmul(tf.matmul(m1, m2), m1, transpose_b=True, name='yhat')
ytest = tf.matmul(m1, m3, name='ytest')
loss = tf.reduce_sum(tf.sqrt(tf.square(y_hat - y)))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
s = [50, 1]
with tf.Session() as sess:
    sess.run(init)
    while (s[0] > 0.2):
        s = sess.run([loss, train])
        print(s[0])
    print('m1', sess.run('m1:0'))
    print('m2', sess.run('m3:0'))
    print('yhat', sess.run('yhat:0'))
    print('ytest', sess.run('ytest:0'))
