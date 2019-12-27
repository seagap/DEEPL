import numpy as np
import tensorflow as tf
def sigmoid(argument):
    return 2
i = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],dtype=tf.float64)
x = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]
              ],dtype=tf.float64)
b1 = tf.get_variable('b1',shape=[10, 1],dtype=tf.float64)
b2 = np.ones([1, 10])
b=tf.matmul(b1,b2)
c = tf.get_variable('c1',shape=[1],dtype=tf.float64)
y = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]
             ,dtype=tf.float64)
yhat=tf.matmul(b,x)+c-tf.square(x)
loss=tf.sqrt(tf.square(yhat-y))
train=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init=tf.global_variables_initializer()
s = [100]
with tf.Session as sess:
    sess.run(init)
    while(s[0]>1):
        s = sess.run([loss, train])
        print("loss", s[0])