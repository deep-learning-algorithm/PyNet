# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:29
# @Author  : zj


from .utils import *
from .im2row import *
from .pool2row import *
from .Layer import *

__all__ = ['Conv2d']


class Conv2d(Layer):
    """
    convolutional layer
    卷积层
    """

    def __init__(self, in_c, filter_h, filter_w, filter_num, stride=1, padding=0, momentum=0, nesterov=False):
        super(Conv2d, self).__init__()
        self.in_c = in_c
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_num = filter_num
        self.stride = stride
        self.padding = padding

        self.W = \
            {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(filter_h * filter_w * in_c, filter_num)),
             'grad': 0,
             'v': 0,
             'momentum': momentum,
             'nesterov': nesterov}
        self.b = {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(1, filter_num)), 'grad': 0}
        self.a = None
        self.input_shape = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.filter_h + 2 * self.padding) / self.stride + 1)
        out_w = int((W - self.filter_w + 2 * self.padding) / self.stride + 1)

        a = im2row_indices(inputs, self.filter_h, self.filter_w, stride=self.stride, padding=self.padding)
        z = a.dot(self.W['val']) + self.b['val']

        self.input_shape = inputs.shape
        self.a = a.copy()

        out = conv_fc2output(z, N, out_h, out_w)
        return out

    def backward(self, grad_out):
        assert len(grad_out.shape) == 4

        dz = conv_output2fc(grad_out)
        self.W['grad'] = self.a.T.dot(dz)
        self.b['grad'] = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        da = dz.dot(self.W['val'].T)
        return row2im_indices(da, self.input_shape, field_height=self.filter_h,
                              field_width=self.filter_w, stride=self.stride, padding=self.padding)

    def update(self, learning_rate=0, regularization_rate=0):
        v_prev = self.W['v']
        self.W['v'] = self.W['momentum'] * self.W['v'] - learning_rate * (
                self.W['grad'] + regularization_rate * self.W['val'])
        if self.W['nesterov']:
            self.W['val'] += (1 + self.W['momentum']) * self.W['v'] - self.W['momentum'] * v_prev
        else:
            self.W['val'] += self.W['v']
        self.b['val'] -= learning_rate * (self.b['grad'])

    def get_params(self):
        return {'W': self.W['val'], 'momentum': self.W['momentum'], 'nesterov': self.W['nesterov'], 'b': self.b['val']}

    def set_params(self, params):
        self.W['val'] = params.get('W')
        self.b['val'] = params.get('b')

        self.W['momentum'] = params.get('momentum', 0.0)
        self.W['nesterov'] = params.get('nesterov', False)
