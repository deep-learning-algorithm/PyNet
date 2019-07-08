# -*- coding: utf-8 -*-

# @Time    : 19-6-30 上午10:12
# @Author  : zj

from .pool2row import *
from .Layer import *

__all__ = ['FC']


class FC:
    """
    fully connected layer
    全连接层
    """

    def __init__(self, num_in, num_out, weight_scale=1e-2):
        """
        :param num_in: 前一层神经元个数
        :param num_out: 当前层神经元个数
        """
        super(FC, self).__init__()
        assert isinstance(num_in, int) and num_in > 0
        assert isinstance(num_out, int) and num_out > 0

        self.num_in = num_in
        self.num_out = num_out
        self.weight_scale = weight_scale

    def __call__(self, inputs, w, b):
        return self.forward(inputs, w, b)

    def forward(self, inputs, w, b):
        # inputs.shape == [N, num_in]
        assert len(inputs.shape) == 2 and inputs.shape[1] == self.num_in
        assert w.shape == (self.num_in, self.num_out)
        assert b.shape == (1, self.num_out)

        z = inputs.dot(w) + b
        cache = (inputs, w, b)
        return z, cache

    def backward(self, grad_out, cache):
        inputs, w, b = cache
        grad_w = inputs.T.dot(grad_out)
        grad_b = np.sum(grad_out, axis=0, keepdims=True) / grad_out.shape[0]

        grad_in = grad_out.dot(w.T)
        return grad_w, grad_b, grad_in

    def get_params(self):
        return self.weight_scale * np.random.normal(loc=0, scale=1.0, size=(self.num_in, self.num_out)), \
               np.zeros((1, self.num_out))
