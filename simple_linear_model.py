# Code from the tutorial TensorFlow Tutorial | Deep Learning Using TensorFlow | Edureka
# https://www.youtube.com/watch?v=yX8KuPZCAMo
# With few correction and addition by Claude COULOMBE - PhD candidate TÉLUQ / UQAM - Montréal

import tensorflow as tf

# Model parameters
W = tf.Variable([0.3],tf.float32)
b = tf.Variable([-0.3],tf.float32)

# Inputs and outputs
x = tf.placeholder(tf.float32)

# Model
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# Loss / cost function
squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)

# Optimize the model
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Initialization
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

for i in range(1001):
	sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))
