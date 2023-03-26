"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import functools
import random


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


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


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):

    '''
    Y = XW + Bias
    Y: n x out
    X: n x in
    W: in x out
    BIas: 1 x out -> n x out
    '''
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features))

        # My current understanding: kaming_unifrom relys on fan_in to decide the up-bound.
        # If we directly init bias through kaiming_uniform(1, self.out_features),
        # the up-bound will be fixed as sqrt(6), which may lack some randomness.
        self.bias = Parameter(init.kaiming_uniform(self.out_features, 1).reshape((1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = X @ self.weight
        Y += self.bias.broadcast_to(Y.shape)
        return Y
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        dim = functools.reduce(lambda a, b: a*b, X.shape[1:])
        new_shape = (X.shape[0], dim)
        return ops.reshape(X, new_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        in_tensor = x
        out_tensor = None
        for module in self.modules:
            out_tensor = module.forward(in_tensor)
            in_tensor = out_tensor
        return out_tensor
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = Parameter(init.one_hot(logits.shape[1], y))
        axes = (1, )
        term1 = ops.logsumexp(logits, axes)
        term2 = (logits * y_one_hot).sum(axes)
        loss = (term1 - term2).sum() / logits.shape[0]
        return loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # weight and bias need to be learned, so pack them as Parameter
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

        # running_mean and running_var are used at test time, no need to be packed as Parameter
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        
        # remember to set the shape of e_x and var_x during computing,
        # but also cache the data for the update of running_mean and running_var
        batch = x.shape[0]
        e_x = x.sum(axes=(0, )) / batch
        minas = x - e_x.broadcast_to(x.shape)
        var_x = (minas ** 2).sum(axes=(0, )) / batch
        std_deviation = (var_x.broadcast_to(x.shape) + self.eps) ** 0.5
        
        if self.training:
            ans = self.weight.broadcast_to(x.shape) \
                * minas / std_deviation \
                + self.bias.broadcast_to(x.shape)

            # avoid unnecessary constructions of graph node
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * e_x.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x.data
            return ans
        else:
            ans = self.weight.broadcast_to(x.shape) \
                * ((x - self.running_mean) / (self.running_var + self.eps) ** 0.5) \
                + self.bias.broadcast_to(x.shape)
            return ans
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        axes = (batch, 1)
        e_x = (x.sum(axes=(1, )) / x.shape[1]).reshape(axes).broadcast_to(x.shape)
        var_x = (((x - e_x) ** 2).sum(axes=(1, )) / x.shape[1]).reshape(axes).broadcast_to(x.shape)
        ans = self.weight.broadcast_to(x.shape) \
            * ((x - e_x) / (var_x + self.eps) ** 0.5) \
            + self.bias.broadcast_to(x.shape)
        return ans
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * init.randb(*x.shape, p=self.p) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
