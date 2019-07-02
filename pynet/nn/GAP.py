# -*- coding: utf-8 -*-

# @Time    : 19-6-21 上午11:18
# @Author  : zj


import numpy as np
from .Layer import *

__all__ = ['GAP']


class GAP(Layer):
    """
    global average pooling layer
    全局平均池化层
    """

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]

        z = np.mean(inputs.reshape(N, C, -1), axis=2)

        cache = (inputs.shape)
        return z, cache

    def backward(self, grad_out, cache):
        input_shape = cache
        N, C, H, W = input_shape[:4]
        dz = grad_out.reshape(N * C, -1)
        da = np.repeat(dz, H * W, axis=1)

        return da.reshape(N, C, H, W)


if __name__ == '__main__':
    gap = GAP()

    inputs = np.arange(36).reshape(2, 2, 3, 3)
    res, cache = gap(inputs)
    print(res)

    grad_out = np.arange(4).reshape(2, 2)
    da = gap.backward(grad_out, cache)
    print(da)
