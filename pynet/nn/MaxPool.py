# -*- coding: utf-8 -*-

# @Time    : 19-7-2 上午10:06
# @Author  : zj

from .utils import *
from .pool2row import *
from .Layer import *

__all__ = ['MaxPool']


class MaxPool:
    """
    max pool layer
    池化层，执行max运算
    """

    def __init__(self, filter_h, filter_w, filter_num, stride=2):
        super(MaxPool, self).__init__()
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_num = filter_num
        self.stride = stride

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.filter_h) / self.stride + 1)
        out_w = int((W - self.filter_w) / self.stride + 1)

        a = pool2row_indices(inputs, self.filter_h, self.filter_w, stride=self.stride)
        z = np.max(a, axis=1)

        arg_z = np.argmax(a, axis=1)
        input_shape = inputs.shape
        a_shape = a.shape
        cache = (arg_z, input_shape, a_shape)

        return pool_fc2output(z, N, out_h, out_w), cache

    def backward(self, grad_out, cache):
        arg_z, input_shape, a_shape = cache

        dz = pool_output2fc(grad_out)
        da = np.zeros(a_shape)
        da[range(a_shape[0]), arg_z] = dz

        return row2pool_indices(da, input_shape, field_height=self.filter_h, field_width=self.filter_w,
                                stride=self.stride)
