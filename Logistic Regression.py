import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()
# for input and output
x = tf.placeholder(tf.float32, shape=[None, 784])
output = tf.placeholder(tf.float32, shape=[None, 10])

# weights and bias units
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
# the engine
y = tf.matmul(x, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output, logits=y))
# train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
for index in range(1500):
    print("epoch " + str(index) + "from 1500")
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], output: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, output: mnist.test.labels}))
