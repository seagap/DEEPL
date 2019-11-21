# coding utf-8
import tensorflow as tf

sess=tf.InteractiveSession()
w = tf.constant("d")
print(w.eval())
sess.close()