"""Optimization module"""
import needle as ndl
import numpy as np
from needle import init


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    '''
    Members:
        lr: learning rate
        momentum: running average parameter
        u: a dict with u[i] representing the grad of the ith parameter that needs to be updated
        weight_decay: l2 regularization parameter
    
    Formula:
        grad := self.weight_decay * w + w.grad
        self.u[i] := self.momentum * self.u[i] + (1 - self.momentum) * grad
        w := w - lr * self.u[i]
        
    Notes:
        1. Avoid unnecessary construction of computaional graph nodes through needle.Tensor.detach() / needle.Tensor.data.
        2. Keep the data type consistent during the entire process.
        3. Do not modify the grad in-place.
    '''
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, w in enumerate(self.params):
            if i not in self.u:
                self.u[i] = init.zeros(*w.shape)
            grad = ndl.Tensor(w.grad.data, dtype=w.dtype).data + self.weight_decay * w.data
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad.data
            w.data = w.data - self.lr * self.u[i]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    '''
    Members:
        lr: learning rate
        beta1: momentum parameter for self.u
        beta2: momentum parameter for self.v
        eps: a small constant used for bias the denominator
        weight_decay: l2 regularization parameter
        u: momentum term of gradient
        v: momentum term of square of gradient
        t: constant used for unbiasing operation, will increment 1 within each step
    
    Formula:
        grad := self.weight_decay * w + w.grad
        self.u[i] := self.beta1 * self.u[i] + (1 - self.beta1) * grad
        self.v[i] := self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
        u_hat = self.u[i] / (1 - self.beta1 ** (t + i))
        v_hat = self.u[i] / (1 - self.beta2 ** (t + i))
        w := w - lr * u_hat / (v_hat ** 0.5 + self.eps)
        
    Notes:
        1. Avoid unnecessary construction of computaional graph nodes through needle.Tensor.detach() / needle.Tensor.data.
        2. Keep the data type consistent during the entire process.
        3. Do not modify the grad in-place.
        4. Remember to unbias the momentum terms.
    '''
    
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

        self.u = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, w in enumerate(self.params):
            if i not in self.u:
                self.u[i] = init.zeros(*w.shape)
                self.v[i] = init.zeros(*w.shape)
            grad = ndl.Tensor(w.grad.data, dtype=w.dtype).data + self.weight_decay * w.data 
            self.u[i] = self.beta1 * self.u[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            u_hat = self.u[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * u_hat / ((v_hat ** 0.5) + self.eps)
        ### END YOUR SOLUTION
