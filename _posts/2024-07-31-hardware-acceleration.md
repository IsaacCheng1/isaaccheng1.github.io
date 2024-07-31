# CPU and GPU Acceleration for Linear Algebra

This doc shows how to build an NDArray from scratch.  


## 1. BackendDevice class (Numpy, CPU or CUDA)
BackendDevice class is a simple wrapper that contains the underlying device backend (e.g., CPU or CUDA).
```python
class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr
```

I implement 3 different backends: 
- a Numpy backend;
- a native CPU backend;
- a GPU backend.

```python
def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]

```

## 2. NDArray class
NDArray class is a generic class that may contain multipe different backends i.e., a Numpy backend, a native CPU backend, or a GPU backend.

```python
class NDArray:
    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle
```

It has 5 fields:
- device - A object of type BackendDevice, which is a simple wrapper that contains a link to the underlying device backend (e.g., CPU or CUDA).
- handle - A class object that stores the underlying memory of the array. This is allocated as a class of type `device.Array()`.
- shape - A tuple specifying the size of each dimension in the array.
- strides - A tuple specifying the strides of each dimension in the array.
- offset - An integer indicating where in the underlying device. Array memory the array actually starts (it's convenient to store this so we can more easily manage pointing back to existing memory, without having to track allocations).


```python
class NDArray:
    ...

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        if prod(new_shape) != prod(self._shape):
            raise ValueError('the product of current shape is not equal to the product of the new shape')
        
        if not self.is_compact():
            raise ValueError('the matrix is not compact')

        return NDArray.make(shape=new_shape, strides=None, device=self._device, handle=self._handle, offset=0)

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        new_shape = tuple([self._shape[i] for i in new_axes])
        new_strides = tuple([self._strides[i] for i in new_axes])

        return self.as_strided(new_shape, new_strides)

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        length_diff = len(new_shape) - len(self._shape)
        new_strides = [0] * length_diff + list(self._strides)

        for idx in range(len(self._shape)):
            if self._shape[idx] != 1:
                assert self._shape[idx] == new_shape[length_diff + idx]
            else:
                new_strides[length_diff + idx] = 0

        return self.as_strided(new_shape, tuple(new_strides))

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        # compute offset
        new_offset = [stride * idx.start for stride, idx in zip(self.strides, idxs)]
        new_offset = sum_up(new_offset)

        # compute strides
        new_strides = [ stride * idx.step for stride, idx in zip(self._strides, idxs) ]
        new_strides = tuple(new_strides)

        # compute shape
        new_shape = [(idx.stop - idx.start - 1) // idx.step + 1 for idx in idxs]
        new_shape = tuple(new_shape)

        return NDArray.make(new_shape, new_strides, self._device, self._handle, new_offset)

    ...
```

A large number of operations likes
- transpose
- broadcast
- sub-setting of matrices
- permute
can all be handled by just adjusting the high-level structure of the array, like it's strides and shape.

