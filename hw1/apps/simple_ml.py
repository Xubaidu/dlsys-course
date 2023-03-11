import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
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
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:

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
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    from needle import log, exp
    from needle import summation
    m = Z.shape[0]
    term1 = log(summation(exp(Z), (1,)))
    term2 = summation(Z * y_one_hot, (1,))
    return summation(term1 - term2) / m
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    iterations = (X.shape[0] + batch - 1) // batch
    for i in range(iterations):
        x_batch = X[i*batch: (i+1)*batch]
        y_batch = y[i*batch: (i+1)*batch]
        
        x = ndl.Tensor(x_batch)
        Z = ndl.relu(x @ W1) @ W2
        
        y_one_hot = np.zeros(Z.shape)
        y_one_hot[np.arange(batch), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()
        
        # The below code snippt will cause the size expansion of computational graph,
        # because we have made W1 a function of lr * W1.grad, and similarly W2 is the same.
        # The expansion will slow down our computing.
        '''
        W1 = W1 - lr * W1.grad
        W2 = W2 - lr * W2.grad
        print(ndl.autograd.TENSOR_COUNTER)
        '''
        
        # We have 2 solutions: one is to use Tensor.detach(),
        # and the other one is to use cached_data of each tensor.
        # I think the 2nd solution is better because it used the cached data.
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
