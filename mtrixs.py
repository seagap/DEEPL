import numpy as np
import tensorflow as tf


def sigmoid(argument):
    return 2


# a1 = tf.Variable(name='a1', initial_value=-10, dtype=tf.float64)
# a2 = tf.Variable(name='a2', initial_value=1, dtype=tf.float64)
# b1 = tf.Variable(name='b1', initial_value=80, dtype=tf.float64)
# b2 = tf.Variable(name='b2', initial_value=-475, dtype=tf.float64)
a1 = tf.Variable(name='a1', initial_value=-1, dtype=tf.float64)
b1 = tf.Variable(name='b1', initial_value=2, dtype=tf.float64)
b2 = tf.Variable(name='b2', initial_value=3, dtype=tf.float64)
x=tf.constant(np.array([[1, 2, 3, 4]
              ]),dtype=tf.float64)
Y=tf.constant(np.array([0,0,1,0]
              ),dtype=tf.float64)

# Yhat = tf.sigmoid(tf.matmul(a, (x ** 2)) - tf.matmul(b, x))
y1=tf.square(x)
y2=a1*y1
y3=b1*x
Yp = y2+y3+b2
Yhat = tf.sigmoid(Yp)
loss = tf.reduce_sum(tf.sqrt(tf.square(Yhat - Y)))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
s = [100]
with tf.Session() as sess:
    sess.run(init)
    while (s[0] > 2.1):
        s = sess.run([loss,Yhat,a1,b1,b2,y1,y2,y3,Yp,train])
        print("loss", s[0],s[2],s[3],s[4])
    print("sigmoid",s[1])
    print("a", s[2])
    print("b", s[3])
    print("c", s[4])
    print("x2", s[5])
    print("ax2", s[6])
    print("bx", s[7])
    print("y", s[8])

