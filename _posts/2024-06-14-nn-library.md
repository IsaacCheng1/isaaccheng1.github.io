# Build a neural network library(simliar as pytorch), from scratch

A neural network library can be composed of:
1. weight initializer;
2. modules which computes an output Tensor given input Tensor(s), such as Residual, SoftmaxLoss;
3. data loader;
4. optimizer

This doc shows the key components of a neural network libray. For more details, please refer to my github [repo](https://github.com/IsaacCheng1/needle/tree/main).

## 1. Weight Initializer
Weight initialization is critial for optimization.  
It really matters.  
A deep network with poorly-chosen weights may never train.  

Here are the weight initializers
```python
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
   std_deviation = gain * math.sqrt(2 / (fan_in + fan_out))
   return randn(fan_in, fan_out, mean=0.0, std=std_deviation, **kwargs)

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std_deviation = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std_deviation, **kwargs)
```

## 2. Modules, which takes Tensor in and computes output Tensor
A module takes a Tensor in and computes output Tensor.  
It has parameter(s), which is trainable Tensor.  

Here is the `Module` class:
```python
def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```
`parameters()` method returns all the parameters of the `Module`.  
`_children()` method returns all the sub-modules (children modules) of the `Module`.  
`__call__` and `forward()` method computes the output Tensor, given input Tensors.  

  
I define various NN Modules below:
- Linear Module
- Residual Module
- Sequential Module
- ReLU Module
- Softmax Loss Module
- Dropout Module
- Normalization Module

### 2.1 Linear Module
```python
class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None

        # initialize W and b with kaiming uniform distribution
        self.weight = Parameter(
                    init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype, requires_grad=True)
        )

        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, requires_grad=True).transpose())

    def forward(self, X: Tensor) -> Tensor:
        output = X @ self.weight
        if self.bias:
            return output + self.bias.broadcast_to(output.shape)
```

### 2.2 Residual Module
```python
class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
```

### 2.3 Sequential Module
```python
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for module in self.modules:
            output = module(output)
        return output
```

### 2.4 ReLU Module
Activation function, such as ReLU, can be designed as `Module` as well.

```python
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
```

### 2.5 Softmax Loss Module
Loss function, such as Softmax Loss, can be designed as `Module` as well.
```python
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # B stands for batch size
        B = logits.shape[0]

        # k stands for # of classes
        k = logits.shape[1]

        # one-hot encoding of Y
        one_hot_Y = init.one_hot(k, y, requires_grad=True) # (B,k)

        loss = ops.logsumexp(logits, axes=(1,)) - (one_hot_Y * logits).sum(axes=(1,))

        return loss.sum() / B
```

### 2.6 Dropout Module
Dropout is a very common way of Regularization.  
It can be designed as `Module` as well.

```python
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # # draw sample from bernoulli distribute
            # # bernoulli is a special distribution of binomial
            # dropout = Tensor(np.random.binomial(n=1, p=1-self.p, size=x.shape)) # element is either 1 or 0

            dropout = init.randb(*x.shape, p=1-self.p, requires_grad=True)
            dropout = dropout * (1 / (1 - self.p))

            return dropout * x
        else:
            return Identity()(x)
```
As you can see, the `forward()` method compute differently for _training_ and _evaluating_.

### 2.7 LayerNorm and BatchNorm
Normalization is critial for optimization.  

It can be designed as `Module` as well.  

```python
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            m, n = x.shape

            # compute the mean over the _batch_ dimension
            x_sum = x.sum(axes=(0,))
            x_mean = x_sum / m # (1,n)

            # compute the running average of mean
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.data # (1, n)

            x_subtract_mean = x - x_mean.broadcast_to(x.shape) # (m,n)

            # compute the variance over the _batch_ dimension
            x_var = x_subtract_mean ** 2  # (m,n)
            x_var = x_var.sum(axes=(0,))  # (1,n)
            x_var = x_var / m # (1,n)

            # compute the running average of variance
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.data # (1, n)

            x_var += self.eps # (1,n)
            x_var = x_var ** (1/2) # (1,n)

            batch_normalized_x = x_subtract_mean / x_var.broadcast_to(x.shape)
            y = batch_normalized_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

            return y
        else:
            running_mean = self.running_mean.broadcast_to(x.shape)
            running_var = self.running_var.broadcast_to(x.shape)

            batch_normalized_x =  (x - running_mean) / ((running_var + self.eps) ** (1 / 2))

            y = batch_normalized_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

            return y

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        m, n = x.shape

        # compute the empirical mean of x
        x_sum = x.sum(axes=(1,))
        x_mean = x_sum / n # (m,)
        x_mean = x_mean.reshape((m, 1))

        x_subtract_mean = x - x_mean.broadcast_to(x.shape) # (m,n)

        # compute the empirical variance of x
        x_var = x_subtract_mean ** 2 # (m,n)
        x_var = x_var.sum(axes=(1,)) # (m,)
        x_var = x_var / n # (m,)
        x_var += self.eps # (m,)
        x_var = x_var ** (1/2) # (m,)
        x_var = x_var.reshape((m, 1))

        normalized_x = x_subtract_mean / x_var.broadcast_to(x.shape)

        y = normalized_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

        return y

```

## 3. Data Loader
`DataLoader` samples mini-batches from the `DataSet`. So, we first need `DataSet` class.    

Here is the `DataSet` class

```python
import struct
import gzip

class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(
            image_filename=image_filename,
            label_filename=label_filename
        )

    def __getitem__(self, index) -> object:
        # note: the index can be a list of integers or just an integer.
        # so, X may be just a ndarray of size 28*28, or a ndarray of 2 dimension (len(index), 28*28)
        X = self.images[index]
        y = self.labels[index]

        X = X.reshape(28, 28, -1) # reshape to (28, 28, len(index))
        X = self.apply_transforms(X)
        X = X.reshape(-1, 28 * 28) # reshape to (len(index), 28*28)
        return X, y

    def __len__(self) -> int:
        return self.labels.size

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

The method `__getitem__` in `DataSet` class returns a sample of the `DataSet` at a given index.  
The `DataLoader` class is different from `DataSet`. It can shuffle the `DataSet`, and sample a minibatch from the `DataSet`.  
Here is the `DataLoader` class:

```python
class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self.ordering):
            raise StopIteration

        samples = [Tensor(x) for x in self.dataset[self.ordering[self.index]]]
        self.index += 1
        return tuple(samples)
```

## 4. Optimizer
Optimizer updates the Parameters. There are different ways to update Parameters. For example,
- weight decay can be applied for regularization (such as L2 Regularization);
- Momentum;
- Adam

Here are the implementations
```python
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            grad = w.grad.realize_cached_data()

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * w.realize_cached_data()

            if self.momentum > 0:
                # keep track of a moving average of multiple previous gradients for grad w.

                # init as 0
                if w not in self.u:
                    self.u[w] = ndl.zeros(*w.shape, requires_grad=False).realize_cached_data()

                self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad

                grad = self.u[w]

            # 1st half is for L2 regularization
            # 2nd half is vanilla SGD
            w.data = w.data - self.lr * grad

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for w in self.params:
            grad = w.grad.realize_cached_data()

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * w.realize_cached_data()

            # keep track of a moving average of multiple previous gradients and square of gradients for grad w.
            # init as 0
            if w not in self.m:
                self.m[w] = ndl.zeros(*w.shape, requires_grad=False).realize_cached_data()
                self.v[w] = ndl.zeros(*w.shape, requires_grad=False).realize_cached_data()

            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)

            # make bias correction
            unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)

            grad = unbiased_m / (unbiased_v ** 0.5 + self.eps)

            w.data = w.data - self.lr * grad
```

## Build a ResNet
Here I use my own NN library to build a ResNet, and train it.

It is worth to mention that the code below is composed of 3 parts:
- `MLPResNet()`;
- `epoch()` function, which takes a DataLoader object and ResNet object and Optimizer object, and return average loss and accuracy.
- `train_mnist()` function, which initialize the DataLoader object, ResNet object and Optimizer object, then call `epoch()`.

```python
def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    module = nn.Residual(
        nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            norm(dim=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=hidden_dim, out_features=dim),
            norm(dim=dim)
        )
    )

    return nn.Sequential(
        module,
        nn.ReLU()
    )

def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
   modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]

   for _ in range(num_blocks):
       modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))

   modules.append(nn.Linear(hidden_dim, num_classes))
   return nn.Sequential(
       *modules
   )

def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    sum_of_loss = 0

    sum_of_hit = 0
    sum_of_examples = 0

    total_num_batches = 0

    loss_function = nn.SoftmaxLoss()

    if opt is not None:
        model.train()

        for i, (X, y) in enumerate(dataloader):
            total_num_batches += 1

            # forward pass
            logits = model(X)

            # reset grad
            opt.reset_grad()

            # compute loss
            loss = loss_function(logits, y)

            # update sum_of_loss
            sum_of_loss += loss.numpy()

            # update sum_of_hit and sum_of_examples
            sum_of_hit += (y.numpy() == np.argmax(logits.numpy(), axis=1)).sum()
            sum_of_examples += y.shape[0]

            # compute grad
            loss.backward()

            # update parameters
            opt.step()

    else:
        model.eval()

        for i, (X, y) in enumerate(dataloader):
            total_num_batches += 1

            # forward pass
            logits = model(X)

            # compute loss
            loss = loss_function(logits, y)

            # update sum_of_loss
            sum_of_loss += loss.numpy()

            # update sum_of_hit and sum_of_examples
            sum_of_hit += (y.numpy() == np.argmax(logits.numpy(), axis=1)).sum()
            sum_of_examples += y.shape[0]

    average_error_rate = (sum_of_examples - sum_of_hit) / sum_of_examples
    average_loss = sum_of_loss / total_num_batches

    return average_error_rate, average_loss


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)

    # initialize the train data loader
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)

    # initialize the test data loader
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # NN
    resNet = MLPResNet(28 * 28,
                       hidden_dim=hidden_dim)

    # initialize optimizer
    opt = optimizer(resNet.parameters(),
                    lr=lr, weight_decay=weight_decay)

    # train
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")

    for i in range(epochs):
        training_accuracy, training_loss = epoch(train_dataloader, resNet, opt)
        test_accuracy, test_loss = epoch(test_dataloader, resNet, None)

        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(i, training_loss, training_accuracy, test_loss, test_accuracy))

    return training_accuracy, training_loss, test_accuracy, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
```

Here are the results:
```
```
