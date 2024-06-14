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


