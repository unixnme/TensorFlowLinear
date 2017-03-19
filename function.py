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

    def linear(self, x):
        """
        given np array x of size (m, n_input),
        returns output y = x*W + b whose size is
        (m, n_output) where m is the # of batches

        given np.array x of size (n_input),
        returns output y = W*x + b whose size is
        (n_output)
        """
        if len(x.shape) == 2:
            assert x.shape[1] == self.n_input
            return np.dot(x, self.W) + self.b

        if len(x.shape) == 1:
            assert x.shape[0] == self.n_input
            return np.dot(x.reshape(1, -1), self.W) + self.b

        else:
            print ("x must be of size (m, n_input) or (n_input)")
