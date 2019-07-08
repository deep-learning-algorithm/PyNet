# -*- coding: utf-8 -*-

# @Time    : 19-6-30 下午4:20
# @Author  : zj


from pynet import nn
from .Net import Net

__all__ = ['ThreeLayerNet']


class ThreeLayerNet(Net):
    """
    实现2层神经网络
    """

    def __init__(self, num_in=0, num_h1=0, num_h2=0, num_out=0, dropout=1.0, weight_scale=1e-2):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.FC(num_in, num_h1, weight_scale=weight_scale)
        self.fc2 = nn.FC(num_h1, num_h2, weight_scale=weight_scale)
        self.fc3 = nn.FC(num_h2, num_out, weight_scale=weight_scale)

        self.relu = nn.ReLU()
        self.z1_cache = None
        self.z1 = None
        self.z2_cache = None
        self.z2 = None
        self.z3_cache = None

        self.use_dropout = dropout != 1.0
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'
            self.dropout_param['p'] = dropout
            self.dropout = nn.Dropout()
            self.U1 = None
            self.U2 = None

        self.params = self._get_params()

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)

        self.z1, self.z1_cache = self.fc1(inputs, self.params['W1'], self.params['b1'])
        a1 = self.relu(self.z1)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            self.U1 = self.dropout(a1.shape, self.dropout_param['p'])
            a1 *= self.U1

        self.z2, self.z2_cache = self.fc2(a1, self.params['W2'], self.params['b2'])
        a2 = self.relu(self.z2)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            self.U2 = self.dropout(a2.shape, self.dropout_param['p'])
            a2 *= self.U2

        z3, self.z3_cache = self.fc3(a2, self.params['W3'], self.params['b3'])

        return z3

    def backward(self, grad_out):
        grad = dict()
        grad['W3'], grad['b3'], da2 = self.fc3.backward(grad_out, self.z3_cache)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            da2 *= self.U2

        dz2 = self.relu.backward(da2, self.z2)
        grad['W2'], grad['b2'], da1 = self.fc2.backward(dz2, self.z2_cache)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            da2 *= self.U2

        dz1 = self.relu.backward(da1, self.z1)
        grad['W1'], grad['b1'], da0 = self.fc1.backward(dz1, self.z1_cache)

        return grad

    def _get_params(self):
        params = dict()
        params['W1'], params['b1'] = self.fc1.get_params()
        params['W2'], params['b2'] = self.fc2.get_params()
        params['W3'], params['b3'] = self.fc3.get_params()
        return params

    def train(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'

    def eval(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'test'
