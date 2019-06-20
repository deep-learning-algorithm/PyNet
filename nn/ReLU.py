# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:34
# @Author  : zj

import numpy as np
from .Layer import *

__all__ = ['ReLU']


class ReLU(Layer):

    def __init__(self):
        super(ReLU, self).__init__()
        self.inputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.inputs = inputs.copy()

        return np.maximum(inputs, 0)

    def backward(self, grad_out):
        assert self.inputs.shape == grad_out.shape

        grad_in = grad_out.copy()
        grad_in[self.inputs < 0] = 0
        return grad_in
