# Softmax Regression NN, from scratch

This doc shows the code to train a _two layer neural network_ on _MNIST_ data.   
The loss function is _softmax / cross-entropy loss_.  
The activation function used in the hidden layer is _ReLU_.   
Use _vanilla_ _SGD_ to update gradient.  

```python
import struct
import numpy as np
import gzip
```

## 1. to load MNIST data
```python
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    X = None
    y = None
    with gzip.open(image_filename) as f:
        images = f.read()

        magic_number = struct.unpack_from('>i', images)
        assert magic_number[0] == 2051

        num_examples = struct.unpack_from('>i', images, offset=4)
        num_examples = num_examples[0]

        num_rows = struct.unpack_from('>i', images, offset=8)
        num_rows = num_rows[0]
        num_cols = struct.unpack_from('>i', images, offset=12)
        num_cols = num_cols[0]
        assert num_rows == 28
        assert num_cols == 28

        fmt = '>%dB' % (num_examples * num_rows * num_cols)
        pixels = struct.unpack_from(fmt, images, offset=16)
        X = np.array(pixels, dtype=np.float32).reshape((num_examples, num_rows * num_cols))
        X = X / 255

    with gzip.open(label_filename) as f:
        labels = f.read()

        magic_number = struct.unpack_from('>i', labels)
        assert magic_number[0] == 2049

        num_examples = struct.unpack_from('>i', labels, offset=4)
        num_examples = num_examples[0]

        fmt = '>%dB' % num_examples
        targets = struct.unpack_from(fmt, labels, offset=8)
        y = np.array(targets, dtype=np.uint8)

    return X, y
```

## 2. to train in 1 epoch
```python
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
    num_examples, input_dim = X.shape
    for i in range(0, num_examples, batch):
        # sample 1 batch
        X_minibatch = X[i: i+batch] # (B,n)
        y_minibatch = y[i: i+batch] # (B,)

        # forward pass
        Z1 = np.matmul(X_minibatch, W1) # (B,d), d is hidden_dim
        Z1 = np.maximum(Z1, 0)
        Z2 = np.matmul(Z1, W2) # (B,k)

        # backward pass
        k = W2.shape[1]
        I_y = np.zeros((batch, k))
        I_y[np.arange(batch), y_minibatch] = 1 # (B, k)

        G2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True) - I_y # (B, k)
        G1 = (Z1 > 0).astype(int) * np.matmul(G2, np.transpose(W2)) # (B,d)

        # SGD
        W1_gradient = np.matmul(np.transpose(X_minibatch), G1) / batch
        W2_gradient = np.matmul(np.transpose(Z1), G2) / batch

        W1 -= lr * W1_gradient
        W2 -= lr * W2_gradient
```

## 3. to train in epochs and test
```python
def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)

    # initialize the weight
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))
```

## 4. run code
```python
if __name__ == "__main__":
    # load data
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    # train
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
```
