# import libraries
import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in Numpy, y = x * 0.1 + 3.0
x_data = np.random.rand(10000).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# We konw that W should be 0.1 and b 0.1 but TensorFlow will figure that out for us
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean square errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(loss)

# Before starting, initialize the variables. we will 'run' this first
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(10001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        
# Learns best fit is W: [0.1], b: [0.3]