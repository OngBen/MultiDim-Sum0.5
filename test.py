import tensorflow as tf
tf.disable_v2_behavior()
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))