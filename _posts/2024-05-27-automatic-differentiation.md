# Automatic Differentiation in python/numpy, from scratch

The way to do automatic differentiation here is:
1. build a computational graph;
2. compute the gradient in the _reverse_ typological order of the computational graph. When computing the gradient, new node is added into the existing computational graph.

Therefore, the method is called as "Reverse Mode Automatic Differentiation by extending the computational graph".

This doc shows the key components of this idea. For more details, please refer to my github [repo](https://github.com/IsaacCheng1/needle/tree/main).

## 1. Build a computational graph
Each node represent an (intermediate) value in the computation. Edges present input output relations.

Here we define a class, `TensorOp`, to represent the operator which operates on input(s) and get output.
```python
import numpy

NDArray = numpy.ndarray

LAZY_MODE = False
TENSOR_COUNTER = 0

class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
```

Here we define a class, `Tensor`, to represent the node.
```python
class Value:
    """A value in the computational graph."""

    # trace of computational graph
    # op and inputs let us know how the Value is computed.
    # for an input node, the op field is None and the inputs field are empty list.
    op: Optional[Op]
    # list of inputs that are fed into the op above
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    # whether to compute the grad wrt. this value
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

```
The critial fields of `Value` class are `op` and `inputs`. `inputs` are fed into the `op` to get the value.
The critial fields of `Tensor` class are `grad`. It is the gradient wrt. the tensor.

## 2. Get topological order of the computational graph
Given a list of nodes at which the computational graph ends, return a topological sort list of nodes.
```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    topo_order = []
    visited = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node.inputs is None:
        topo_order.append(node)
        return

    for predecessor in node.inputs:
        if predecessor not in visited:
            visited.add(predecessor)
            topo_sort_dfs(predecessor, visited, topo_order)
    topo_order.append(node)
```

## 3. Compute gradient in the _reverse_ typological order of the computational graph
Traverse the computational graph reversely.  
During traversal, compute gradient and extend the computational graph.
```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output tensor with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_tensor)
    # instead of the vector output_tensor. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_tensor that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node_j in reverse_topo_order:
        # sum up the partial ad-joints (i.e. the gradient contributions from each output node)
        node_j.grad = sum_node_list(node_to_output_grads_list[node_j])

        if node_j.op is None:
            continue

        partial_adjoints = node_j.op.gradient_as_tuple(node_j.grad, node_j)
        for node_i, partial_adjoint in zip(node_j.inputs, partial_adjoints):
            if node_i not in node_to_output_grads_list:
                node_to_output_grads_list[node_i] = []
            node_to_output_grads_list[node_i].append(partial_adjoint)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
```

## 4. train 2 layer Softmax Regression NN
```python
def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
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
    m, _ = Z.shape
    loss = ndl.log(ndl.exp(Z).sum(axes=(1,))) - (Z * y_one_hot).sum(axes=(1,))
    return ndl.summation(loss) / m

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
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
    num_examples, input_dim = X.shape
    _, num_classes = W2.shape
    for i in range(0, num_examples, batch):
        # sample 1 batch
        X_minibatch = X[i: i+batch] # (B,n)
        y_minibatch = y[i: i+batch] # (B,)
        y_one_hot = np.zeros((batch, num_classes)) # (B,K)
        y_one_hot[np.arange(batch), y_minibatch] = 1

        X_tensor = ndl.Tensor(X_minibatch)
        y_tensor = ndl.Tensor(y_one_hot)

        # forward pass (construct computational graph)
        Z = ndl.relu(X_tensor @ W1) @ W2 # (B,K)
        loss = softmax_loss(Z, y_tensor) # scalar

        # compute gradient
        loss.backward()

        # SGD
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return (W1, W2)
```

## 5. Training Result
```
| Epoch | Train Loss | Train Err | Test Loss | Test Err |
|     0 |    0.19770 |   0.06010 |   0.19683 |  0.05850 |
|     1 |    0.14165 |   0.04313 |   0.14988 |  0.04490 |
|     2 |    0.11250 |   0.03378 |   0.12793 |  0.03920 |
|     3 |    0.09454 |   0.02820 |   0.11635 |  0.03640 |
|     4 |    0.08189 |   0.02443 |   0.10848 |  0.03390 |
|     5 |    0.07224 |   0.02158 |   0.10271 |  0.03150 |
|     6 |    0.06472 |   0.01957 |   0.09812 |  0.03050 |
|     7 |    0.05819 |   0.01743 |   0.09513 |  0.02920 |
|     8 |    0.05271 |   0.01543 |   0.09283 |  0.02890 |
|     9 |    0.04814 |   0.01398 |   0.09123 |  0.02780 |
|    10 |    0.04384 |   0.01240 |   0.08945 |  0.02700 |
|    11 |    0.04068 |   0.01150 |   0.08854 |  0.02720 |
|    12 |    0.03740 |   0.01033 |   0.08722 |  0.02680 |
|    13 |    0.03465 |   0.00957 |   0.08638 |  0.02530 |
|    14 |    0.03209 |   0.00877 |   0.08560 |  0.02570 |
|    15 |    0.02958 |   0.00788 |   0.08481 |  0.02450 |
|    16 |    0.02773 |   0.00725 |   0.08469 |  0.02490 |
|    17 |    0.02576 |   0.00662 |   0.08403 |  0.02490 |
|    18 |    0.02419 |   0.00590 |   0.08405 |  0.02470 |
|    19 |    0.02257 |   0.00548 |   0.08338 |  0.02470 |
```

