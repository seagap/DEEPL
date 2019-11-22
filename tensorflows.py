import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


boston = tf.contrib.learn.datasets.load_datasset('boston')
X_train, Y_train = boston.data[:, 5], boston.target
n_samples = len(X_train)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

b = tf.Variable(0.0)
w = tf.Variable(0.0)

Y_hat = X * w + b
loss = tf.square(Y - Y_hat, name='loss')
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
init_op = tf.global_variables_initializer()
total = []

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('graphs', sess.graph)
    for i in range(100):
        total_loss = 0
        for x, y in zip(X_train, Y_train):
            result = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += 1
        total.append(total_loss / n_samples)
        print('Epoch{0}:Loss{1}'.format(i, total_loss / n_samples))
    writer.close()
    b_value, w_value = sess.run([b, w])
Y_pred = X_train * w_value + b_value
print('Done')
plt.plot(X_train, Y_train, 'bo', label='Real Data')
plt.plot(X_train, Y_train, 'r', label='Predicated Data')
plt.legend()
plt.show()
plt.plot(total)
plt.show()
#一对一的x和y 所以没有网络
