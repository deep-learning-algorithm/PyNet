# -*- coding: utf-8 -*-

# @Time    : 19-6-30 上午10:33
# @Author  : zj

import numpy as np
from .Layer import *

__all__ = ['ReLU']


class ReLU(object):

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        return np.maximum(inputs, 0)

    def backward(self, grad_out, inputs):
        assert inputs.shape == grad_out.shape

        grad_in = grad_out.copy()
        grad_in[inputs < 0] = 0
        return grad_in
