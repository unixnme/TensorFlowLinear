import tensorflow as tf
import numpy as np

class Function(object):
    """
    this class contains the unknown function to look for
    """
    def __init__(self, n_input, n_output):
        """
        defines the input size and output size
        of the function

        it initializes the function with random weight
        and bias
        """
        self.n_input = n_input
        self.n_output = n_output

        self.W = np.random.rand(n_input, n_output)
        self.b = np.random.rand(n_output)

    def calc(self, x):
        """
        given np array x of size (m, n_input),
        returns output y = x*W + b whose size is
        (m, n_output) where m is the # of batches

        given np.array x of size (n_input),
        returns output y = W*x + b whose size is
        (n_output)
        """
        if len(x.shape) == 2:
            assert x.shape[1] == n_input
            return np.dot(x, self.W) + self.b

        if len(x.shape) == 1:
            assert x.shape[0] == n_input
            return np.dot(x.reshape(1, -1), self.W) + self.b

        else:
            print ("x must be of size (m, n_input) or (n_input)")

n_input = 10
n_output = 5

f = Function(n_input, n_output)

x = tf.placeholder(tf.float32, (None, n_input))
y = tf.placeholder(tf.float32, (None, n_output))

mu = 0
sigma = 0.1
learn_rate = 0.01

W = tf.Variable(tf.truncated_normal([n_input, n_output], mu, sigma))
b = tf.Variable(tf.truncated_normal([n_output], mu, sigma))

layer1 = tf.matmul(x, W) + b
loss = tf.reduce_mean((layer1 - y)**2)
optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
training_operation = optimizer.minimize(loss)

EPOCHS = 1000
BATCH_SIZE = 5

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        X_train = np.random.rand(BATCH_SIZE, n_input)
        Y_train = f.calc(X_train)
        session.run(training_operation, feed_dict={x:X_train, y:Y_train})
        score = session.run(loss, feed_dict={x:X_train, y:Y_train})
        print ("epoch " + str(i+1) + ": " + str(score))

    X_test = np.random.rand(BATCH_SIZE, n_input)
    Y_test = f.calc(X_test)
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
    

