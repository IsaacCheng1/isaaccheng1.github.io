# Softmax Regression, from scratch

## 1. load data
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
