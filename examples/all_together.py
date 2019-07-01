# -*- coding: utf-8 -*-

# @Time    : 19-7-1 上午10:06
# @Author  : zj

import numpy as np
from pynet.vision.data import mnist
import pynet.nn as nn
from pynet.solver import *


class ThreeLayerNet3:
    """
    实现2层神经网络
    """

    def __init__(self, num_in=0, num_h1=0, num_h2=0, num_out=0, weight_scale=1e-2):
        super(ThreeLayerNet3, self).__init__()
        self.params = dict()
        self.params['W1'] = weight_scale * np.random.randn(num_in, num_h1)
        self.params['b1'] = weight_scale * np.random.randn(1, num_h1)
        self.params['W2'] = weight_scale * np.random.randn(num_h1, num_h2)
        self.params['b2'] = weight_scale * np.random.randn(1, num_h2)
        self.params['W3'] = weight_scale * np.random.randn(num_h2, num_out)
        self.params['b3'] = weight_scale * np.random.randn(1, num_out)

        self.a1_cache = None
        self.a1 = None
        self.a3_cache = None
        self.a3 = None
        self.a5_cache = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, D_in]
        assert len(inputs.shape) == 2
        self.a1, self.a1_cache = affine_forward(inputs, self.params['W1'], self.params['b1'])
        z2 = relu_forward(self.a1)
        self.a3, self.a3_cache = affine_forward(z2, self.params['W2'], self.params['b2'])
        z4 = relu_forward(self.a3)
        a5, self.a5_cache = affine_forward(z4, self.params['W3'], self.params['b3'])

        return a5

    def backward(self, grad_out):
        grad = dict()
        grad['W3'], grad['b3'], dz4 = nn.affine_backward(grad_out, self.a5_cache)
        da3 = relu_backward(dz4, self.a3)
        grad['W2'], grad['b2'], dz2 = nn.affine_backward(da3, self.a3_cache)
        da1 = relu_backward(dz2, self.a1)
        grad['W1'], grad['b1'], da0 = nn.affine_backward(da1, self.a1_cache)

        return grad


def relu_forward(inputs):
    return np.maximum(inputs, 0)


def relu_backward(grad_out, inputs):
    assert inputs.shape == grad_out.shape

    grad_in = grad_out.copy()
    grad_in[inputs < 0] = 0
    return grad_in


def affine_forward(inputs, w, b):
    res = inputs.dot(w) + b
    cache = (inputs, w, b)
    return res, cache


def affine_backward(grad_out, cache):
    inputs, w, b = cache

    grad_w = inputs.T.dot(grad_out)
    grad_b = np.sum(grad_out, axis=0, keepdims=True) / grad_out.shape[0]
    grad_in = grad_out.dot(w.T)
    return grad_w, grad_b, grad_in


data_path = '/home/zj/data/decompress_mnist'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = mnist.load_mnist(data_path, shuffle=True, is_flatten=True)

    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_test,
        'y_val': y_test
    }
    print(data.keys())
    print(x_train.shape)
    print(y_train.shape)

    model = ThreeLayerNet3(num_in=784, num_h1=1200, num_h2=200, num_out=10)
    criterion = nn.CrossEntropyLoss()

    solver = Solver(model, data, criterion, update_rule='sgd', batch_size=256, num_epochs=100, reg=1e-3,
                    optim_config={'learning_rate': 1e-3})
    solver.train()
