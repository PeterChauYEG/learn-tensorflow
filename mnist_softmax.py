"""import data and assign it to a varible"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""import tensorflow"""
import tensorflow as tf

"""placeholder - declare symbolic variables"""
"""2-D tensor of floating-point numbers with a shape of [NONE, 784]"""
x = tf.placeholder(tf.float32, [None, 784])

"""declare model parameters as Variables which lives in TensorFlow's graph of interacting operations"""
"""initialize W and b tensors with zeros"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""implement our model"""
y = tf.nn.softmax(tf.matmul(x, W) + b)


"""training"""

"""implement cross-entropy to determine the loss of our model"""
"""declare placeholder to input correct answers"""
y_ = tf.placeholder(tf.float32, [None, 10])

"""implement the cross-entropy function"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""use gradient descent algorithm to modify variables and reduce loss"""
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""define initialization of the variables we created"""
init = tf.initialize_all_variables()

"""launch the model in a session and initialize the variables"""
sess = tf.Session()
sess.run(init)

"""run the training step 1000 times"""
for i in range(1000):
  """get a batch of 100 random data points"""
  batch_xs, batch_ys = mnist.train.next_batch(100)
  """feed batch into train_step to replace placeholders"""
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
"""evaluating the model"""
"""compare the label our model thinks is most likely for each input to the correct label"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

"""cast correct_prediction to floating-point numbers then take the mean"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""print accuracy on our test data"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))