import tensorflow as tf
import numpy as np
from function import *

n_input = 10
n_output = 5

f = Function(n_input, n_output)

x = tf.placeholder(tf.float32, (None, n_input))
y = tf.placeholder(tf.float32, (None, n_output))
r = tf.placeholder(tf.float32)

mu = 0
sigma = 0.1
learn_rate = 0.01

W = tf.Variable(tf.truncated_normal([n_input, n_output], mu, sigma))
b = tf.Variable(tf.truncated_normal([n_output], mu, sigma))

layer1 = tf.matmul(x, W) + b
loss = tf.reduce_mean((layer1 - y)**2)
optimizer = tf.train.AdamOptimizer(learning_rate = r)
training_operation = optimizer.minimize(loss)

EPOCHS = 5000
BATCH_SIZE = 5

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        X_train = np.random.rand(BATCH_SIZE, n_input)
        Y_train = f.linear(X_train)
        score = session.run(loss, feed_dict={x:X_train, y:Y_train})
        print ("epoch " + str(i+1) + ": " + str(score))
        session.run(training_operation, feed_dict={x:X_train, y:Y_train, r:learn_rate})

    X_test = np.random.rand(BATCH_SIZE, n_input)
    Y_test = f.linear(X_test)
    print()
    print ("approximated y:")
    print (session.run(layer1, feed_dict={x:X_test, y:Y_test}))
    print()
    print ("true y:")
    print (Y_test)
    print()

    print ("approximated weight")
    print(session.run(W))
    print()
    print ("true weight")
    print(f.W)
    print()
    print ("approximated bias")
    print(session.run(b))
    print()
    print ("true bias")
    print(f.b)
    

