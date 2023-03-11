import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:

        # MNIST stores data in big-endian format
        # load header variables
        _ = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        # load images pixels
        images_data = f.read(num_images * num_rows * num_cols)
        images = struct.unpack('>' + 'B' * len(images_data), images_data)

        # reshape images
        images = np.array(images).reshape(num_images, num_rows * num_cols)

        # normalize the dataset
        images = images / 255

        # convert the data to fp32
        images = images.astype(np.float32)

    with gzip.open(label_filename, 'rb') as f:

        # load header variables
        _ = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        # load images pixels
        labels_data = f.read(num_labels)
        labels = struct.unpack('>' + 'B' * len(labels_data), labels_data)

        # reshape images
        labels = np.array(labels)

        # convert the data to uint8
        labels = labels.astype(np.uint8)

    return (images, labels)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.sum(-Z[np.arange(Z.shape[0]), y] + np.log(np.sum(np.exp(Z), axis=1))) / len(y)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE

    # m: the num of testcase
    # n: the num of features
    # k: the num of labels
    # X:     mxn
    # y:     mx1
    # theta: nxk
    # gradient as to theta: nxk

    # divide testset into different epoches
    iterations = (y.shape[0] + batch - 1) // batch
    for i in range(iterations):

        # theta := theta - lr / batch * sum(gradient)
        # gradient = X^T * (Z - Iy)
        batched_X = X[i*batch: (i+1)*batch]
        batched_y = y[i*batch: (i+1)*batch]

        # Z = normalize(exp(X * theta)), mxk dim
        Z = np.exp(batched_X @ theta)
        Z = Z / np.sum(Z, axis=1, keepdims=True)

        # Iy, Iy[i][y_i] = 1, others are 0, mxk dim
        Iy = np.zeros(Z.shape)
        Iy[np.arange(Z.shape[0]), batched_y] = 1

        # the sum of gradient, nxk dim
        grad = batched_X.T @ (Z - Iy)
        theta -= grad * (lr / batch)

    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    
    def relu(x):
        return np.where(x > 0, x, 0)
    
    def d_relu(x):
        return np.where(x > 0, 1, 0)
    
    def identification(x):
        return x
    
    def d_identification(x):
        return np.ones(x.shape)
    
    def calc_gradient_bf(X, y, W1, W2):
        """ Derive the gradient w.r.t W1 and W2 by handwritten respectively.
        Calculate them directly. Not a general method.
        """
        
        # S = normalize(exp(sigma(X @ W1) @ W2))
        S = np.exp(relu(X @ W1) @ W2)
        S = S / np.sum(S, axis=1, keepdims=True)
        
        Iy = np.zeros(S.shape)
        Iy[np.arange(S.shape[0]), y] = 1
        
        gradient_w1 = np.transpose(xx) @ ((S - Iy) @ np.transpose(W2) * d_relu(X @ W1))
        gradient_w2 = np.transpose(relu(xx @ W1)) @ (S - Iy)
        
        return [
            gradient_w1,
            gradient_w2,
        ]
        
    def calc_gradient(L, X, y, W):
        
        """ Do the backpropagation iteratively. The time complexity is O(L).
        Will modify W in place.
        
        Args:
            L (int): the num of layers
            X (np.ndarray[np.float32]):
                2D tensor, the sampled dataset, batch x input_dim
            y (np.ndarray[np.uint8]):
                1D tensor, the sampled labelset, batch x 1
            W (list[np.ndarray[np.float32]]):
                the list of parameters for each layer
        """
        
        # define sigma functions and their derivatives
        sigma = [
            relu,
            identification,
        ]
        d_sigma = [
            d_relu,
            d_identification,
        ]
        
        # Z[i]: output of layer i
        # Z[i + 1] = sigma[i](Z[i] @ W[i])
        # i = 0, 1, ..., L - 1
        # Z[1] = x
        Z = [X] + [None] * (L)
        for i in range(0, L):
            Z[i + 1] = sigma[i](Z[i] @ W[i])
            
        # S = normalize(exp(sigma(X @ W1) @ W2))
        S = np.exp(Z[L])
        S = S / np.sum(S, axis=1, keepdims=True)
        
        Iy = np.zeros(S.shape)
        Iy[np.arange(S.shape[0]), y] = 1
        
        # G[i]: cached terms of partial derivatives
        # G[i] = (G[i + 1] * d_sigma[i](Z[i] @ W[i])) @ transpose(Z[i])
        # i = 0, 1, ..., L
        # G[L + 1] = S - Iy
        G = [None] * L + [S - Iy]
        for i in range(L - 1, -1, -1):
            G[i] = (G[i + 1] * d_sigma[i](Z[i] @ W[i])) @ np.transpose(W[i])
        
        # gradient_w_i = transpose(Z[i]) @ (G[i + 1] * d_sigma[i](Z[i] @ W[i]))
        gradients = [None] * L
        for i in range(L - 1, -1, -1):
            gradients[i] = np.transpose(Z[i]) @ (G[i + 1] * d_sigma[i](Z[i] @ W[i]))
            
        return gradients
    
    iterations = (X.shape[0] + batch - 1) // batch
    for i in range(iterations):
        xx = X[i*batch: (i+1)*batch]
        yy = y[i*batch: (i+1)*batch]
        layer_num = 2
        
        gradients_bf = calc_gradient_bf(X=xx, y=yy, W1=W1, W2=W2)
        gradients = calc_gradient(L=layer_num, X=xx, y=yy, W=[W1, W2])

        assert len(gradients_bf)==len(gradients), 'the length of the calculated gradient should be the same'
        for i in range(len(gradients)):
            assert (gradients_bf[i] == gradients[i]).all(), f'calculated gradient should be the same, but fail at gradient {i}'

        W = [W1, W2]
        for i in range(layer_num):
            W[i] -= lr / batch * gradients[i]
        
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
