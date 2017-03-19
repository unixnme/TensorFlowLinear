import tensorflow as tf
import numpy as np
from function import *

n_input = 3
n_hidden = 100
f = Trig(n_input)

x = tf.placeholder(tf.float32, (None, n_input))
y = tf.placeholder(tf.float32, (None, n_input))
r = tf.placeholder(tf.float32)

mu = 0
sigma = 0.1
learn_rate = 0.001

W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], mu, sigma))
b1 = tf.Variable(tf.truncated_normal([n_hidden], mu, sigma))

layer1 = tf.matmul(x, W1) + b1
layer1 = tf.sigmoid(layer1)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_input], mu, sigma))
b2 = tf.Variable(tf.truncated_normal([n_input], mu, sigma))

layer2 = tf.matmul(layer1, W2) + b2

loss = tf.reduce_mean((layer2 - y)**2)
optimizer = tf.train.AdamOptimizer(learning_rate = r)
training_operation = optimizer.minimize(loss)

EPOCHS = 30000
BATCH_SIZE = 10

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        X_train = np.random.rand(BATCH_SIZE, n_input)
        Y_train = f.sin(X_train)
        score = session.run(loss, feed_dict={x:X_train, y:Y_train})
        print ("epoch " + str(i+1) + ": " + str(score))
        session.run(training_operation, feed_dict={x:X_train, y:Y_train, r:learn_rate})

    X_test = np.random.rand(BATCH_SIZE, n_input)
    Y_test = f.sin(X_test)

    print()
    print ("diff in y:")
    print (Y_test - session.run(layer2, feed_dict={x:X_test, y:Y_test}))
    print()
