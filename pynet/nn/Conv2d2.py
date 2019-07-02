# -*- coding: utf-8 -*-

# @Time    : 19-7-2 上午9:53
# @Author  : zj


from .utils import *
from .im2row import *
from .pool2row import *
from .Layer import *

__all__ = ['Conv2d2']


class Conv2d2:
    """
    convolutional layer
    卷积层
    """

    def __init__(self, in_c, filter_h, filter_w, filter_num, stride=1, padding=0, weight_scale=1e-2):
        """
        :param in_c: 输入数据体通道数
        :param filter_h: 滤波器长
        :param filter_w: 滤波器宽
        :param filter_num: 滤波器个数
        :param stride: 步长
        :param padding: 零填充
        :param weight_scale:
        """
        super(Conv2d2, self).__init__()
        self.in_c = in_c
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_num = filter_num
        self.stride = stride
        self.padding = padding
        self.weight_scale = weight_scale

    def __call__(self, inputs, w, b):
        return self.forward(inputs, w, b)

    def forward(self, inputs, w, b):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.filter_h + 2 * self.padding) / self.stride + 1)
        out_w = int((W - self.filter_w + 2 * self.padding) / self.stride + 1)

        a = im2row_indices(inputs, self.filter_h, self.filter_w, stride=self.stride, padding=self.padding)
        z = a.dot(w) + b

        out = conv_fc2output(z, N, out_h, out_w)
        cache = (a, inputs.shape, w, b)
        return out, cache

    def backward(self, grad_out, cache):
        assert len(grad_out.shape) == 4

        a, input_shape, w, b = cache

        dz = conv_output2fc(grad_out)
        grad_W = a.T.dot(dz)
        grad_b = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        da = dz.dot(w.T)
        return grad_W, grad_b, row2im_indices(da, input_shape, field_height=self.filter_h,
                                              field_width=self.filter_w, stride=self.stride, padding=self.padding)

    def get_params(self):
        return self.weight_scale * np.random.normal(loc=0, scale=1.0, size=(
            self.filter_h * self.filter_w * self.in_c, self.filter_num)), \
               self.weight_scale * np.random.normal(loc=0, scale=1.0, size=(1, self.filter_num))
