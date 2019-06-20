# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:03
# @Author  : zj


from .utils import *
from .pool2row import *
from .Layer import *

__all__ = ['MaxPool']


class MaxPool(Layer):
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

        self.input_shape = None
        self.a_shape = None
        self.arg_z = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.filter_h) / self.stride + 1)
        out_w = int((W - self.filter_w) / self.stride + 1)

        a = pool2row_indices(input, self.filter_h, self.filter_w, stride=self.stride)
        z = np.max(a, axis=1)
        self.arg_z = np.argmax(a, axis=1)
        self.input_shape = inputs.shape
        self.a_shape = a.shape

        return pool_fc2output(z, N, out_h, out_w)

    def backward(self, grad_out):
        dz = pool_output2fc(grad_out)
        da = np.zeros(self.a_shape)
        da[range(self.a_shape[0]), self.arg_z] = dz

        return row2pool_indices(da, self.input_shape, field_height=self.filter_h, field_width=self.filter_w,
                                stride=self.stride)
