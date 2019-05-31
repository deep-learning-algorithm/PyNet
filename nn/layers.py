# -*- coding: utf-8 -*-

# @Time    : 19-5-26 下午8:10
# @Author  : zj

import numpy as np
from abc import ABCMeta, abstractmethod
from nn.im2row import *
from nn.pool2row import *
from nn.layer_utils import *


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass


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
        self.nesterov = nesterov

        self.W = \
            {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(filter_h * filter_w * in_c, filter_num)),
             'grad': 0,
             'v': 0,
             'momentum': momentum}
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
        if self.nesterov:
            self.W['val'] += (1 + self.W['momentum']) * self.W['v'] - self.W['momentum'] * v_prev
        else:
            self.W['val'] += self.W['v']
        self.b['val'] -= learning_rate * (self.b['grad'])

    def get_params(self):
        return {'W': self.W['val'], 'b': self.b['val']}

    def set_params(self, params):
        self.W['val'] = params['W']
        self.b['val'] = params['b']


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

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        # input.shape == [N, C, H, W]
        assert len(input.shape) == 4
        N, C, H, W = input.shape[:4]
        out_h = int((H - self.filter_h) / self.stride + 1)
        out_w = int((W - self.filter_w) / self.stride + 1)

        a = pool2row_indices(input, self.filter_h, self.filter_w, stride=self.stride)
        z = np.max(a, axis=1)
        self.arg_z = np.argmax(a, axis=1)
        self.input_shape = input.shape
        self.a_shape = a.shape

        return pool_fc2output(z, N, out_h, out_w)

    def backward(self, grad_out):
        dz = pool_output2fc(grad_out)
        da = np.zeros(self.a_shape)
        da[range(self.a_shape[0]), self.arg_z] = dz

        return row2pool_indices(da, self.input_shape, field_height=self.filter_h, field_width=self.filter_w,
                                stride=self.stride)


class FC(Layer):
    """
    fully connected layer
    全连接层
    """

    def __init__(self, num_in, num_out, momentum=0, nesterov=False):
        """
        :param num_in: 前一层神经元个数
        :param num_out: 当前层神经元个数
        :param momentum: 动量因子
        :param nesterov: 是否使用Nesterov加速梯度
        """
        super(FC, self).__init__()
        assert isinstance(num_in, int) and num_in > 0
        assert isinstance(num_out, int) and num_out > 0

        self.nesterov = nesterov
        self.W = {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(num_in, num_out)),
                  'grad': 0,
                  'v': 0,
                  'momentum': momentum}
        self.b = {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(1, num_out)), 'grad': 0}
        self.inputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape == [N, num_in]
        assert len(inputs.shape) == 2
        self.inputs = inputs.copy()

        z = inputs.dot(self.W['val']) + self.b['val']
        return z

    def backward(self, grad_out):
        self.W['grad'] = self.inputs.T.dot(grad_out)
        self.b['grad'] = np.sum(grad_out, axis=0, keepdims=True) / grad_out.shape[0]

        grad_in = grad_out.dot(self.W['val'].T)
        return grad_in

    def update(self, learning_rate=0, regularization_rate=0):
        v_prev = self.W['v']
        self.W['v'] = self.W['momentum'] * self.W['v'] - learning_rate * (
                self.W['grad'] + regularization_rate * self.W['val'])
        if self.nesterov:
            self.W['val'] += (1 + self.W['momentum']) * self.W['v'] - self.W['momentum'] * v_prev
        else:
            self.W['val'] += self.W['v']
        self.b['val'] -= learning_rate * self.b['grad']

    def get_params(self):
        return {'W': self.W['val'], 'b': self.b['val']}

    def set_params(self, params):
        self.W['val'] = params['W']
        self.b['val'] = params['b']


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


class CrossEntropyLoss(object):

    def __init__(self):
        self.probs = None
        self.labels = None

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        # scores.shape == [N, grad_out]
        # labels.shape == [N]
        assert len(scores.shape) == 2
        assert len(labels.shape) == 1
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        self.probs = expscores / np.sum(expscores, axis=1, keepdims=True)
        self.labels = labels.copy()

        N = labels.shape[0]
        correct_probs = self.probs[range(N), labels]
        loss = -1.0 / N * np.sum(np.log(correct_probs))
        return loss

    def backward(self):
        # grad_out = probs - Y
        grad_out = self.probs
        N = self.labels.shape[0]

        grad_out[range(N), self.labels] -= 1
        return grad_out

    def get_probs(self):
        return self.probs


class Softmax(object):
    """
    softmax评分
    """

    def __init__(self):
        pass

    def __call__(self, scores):
        return self.forward(scores)

    def forward(self, scores):
        # scores.shape == [N, C]
        assert len(scores.shape) == 2
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        probs = expscores / np.sum(expscores, axis=1, keepdims=True)

        return probs
