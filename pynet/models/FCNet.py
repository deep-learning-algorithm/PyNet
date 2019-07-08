# -*- coding: utf-8 -*-

# @Time    : 19-7-8 下午1:36
# @Author  : zj

"""
参考 cs231n assignment2 FullyConnectedNet，实现自定义层数和大小的神经网络

网络结构为

{FC - [batch/layer norm] - RELU - [dropout]} * (L - 1) - FC
"""

__all__ = ['FCNet']

import numpy as np
from pynet import nn
from .Net import Net


class FCNet(Net):
    """
    神经网络
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10, normalization=None,
                 dropout=1.0, seed=None, weight_scale=1e-2, dtype=np.double):
        super(FCNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.normalization = normalization
        self.weight_scale = weight_scale
        self.dtype = dtype

        self.use_dropout = dropout != 1
        self.num_layers = 1 + len(hidden_dims)
        self.relu = nn.ReLU()

        self.fcs = self._get_fcs()
        self.params = self._get_params()
        self.caches = self._get_caches()

        self.use_dropout = dropout != 1.0
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout = nn.Dropout()

            self.dropout_param['mode'] = 'train'
            self.dropout_param['p'] = dropout
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn = nn.BN()

            for i in range(len(hidden_dims)):
                self.params['gamma%d' % (i + 1)] = np.ones(hidden_dims[i])
                self.params['beta%d' % (i + 1)] = np.zeros(hidden_dims[i])
                self.bn_params.append({'mode': 'train'})

        # 转换可学习参数为指定数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)
        inputs = inputs.astype(self.dtype)

        x = None
        for i in range(self.num_layers):
            w = self.params['W%d' % (i + 1)]
            b = self.params['b%d' % (i + 1)]

            if i == 0:
                x = inputs
            self.caches['z%d' % (i + 1)], self.caches['z%d_cache' % (i + 1)] = self.fcs[i].forward(x, w, b)

            if i != (self.num_layers - 1):
                z = self.caches['z%d' % (i + 1)]

                if self.normalization == 'batchnorm':
                    gamma = self.params['gamma%d' % (i + 1)]
                    beta = self.params['beta%d' % (i + 1)]
                    bn_param = self.bn_params[i]

                    y, self.caches['y%d_cache' % (i + 1)] = self.bn(z, gamma, beta, bn_param)
                    z = y

                x = self.relu(z)

                if self.use_dropout:
                    x, self.caches['dropout%d_cache' % (i + 1)] = self.dropout(x, self.dropout_param)

        return self.caches['z%d' % self.num_layers]

    def backward(self, grad_out):
        grad = dict()

        da = None
        for i in reversed(range(self.num_layers)):
            z = self.caches['z%d' % (i + 1)]
            z_cache = self.caches['z%d_cache' % (i + 1)]

            if i == (self.num_layers - 1):
                dz = grad_out
            else:
                if self.use_dropout:
                    dropout_cache = self.caches['dropout%d_cache' % (i + 1)]
                    da = self.dropout.backward(da, dropout_cache)

                dz = self.relu.backward(da, z)

                if self.normalization == 'batchnorm':
                    y_cache = self.caches['y%d_cache' % (i + 1)]
                    dy = dz
                    dz, grad['gamma%d' % (i + 1)], grad['beta%d' % (i + 1)] = self.bn.backward(dy, y_cache)

            grad['W%d' % (i + 1)], grad['b%d' % (i + 1)], da = self.fcs[i].backward(dz, z_cache)

        self.caches = self._get_caches()
        return grad

    def _get_fcs(self):
        fcs = list()
        if self.hidden_dims is None:
            fcs.append(nn.FC(self.input_dim, self.num_classes, weight_scale=self.weight_scale))
        else:
            for i in range(self.num_layers):
                if i == 0:
                    num_in = self.input_dim
                else:
                    num_in = self.hidden_dims[i - 1]

                if i == (self.num_layers - 1):
                    num_out = self.num_classes
                else:
                    num_out = self.hidden_dims[i]

                fcs.append(nn.FC(num_in, num_out))

        return fcs

    def _get_params(self):
        params = dict()
        for i, fc in enumerate(self.fcs):
            params['W%d' % (i + 1)], params['b%d' % (i + 1)] = fc.get_params()
        return params

    def _get_caches(self):
        caches = dict()
        for i in range(1, self.num_layers):
            caches['z%d' % i] = None
            caches['z%d_cache' % i] = None
            if i != (self.num_layers - 1):
                if self.normalization == 'batchnorm':
                    caches['y%d_cache' % (i + 1)] = None
                if self.use_dropout:
                    caches['dropout%d_cache' % (i + 1)] = None

        return caches

    def train(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'
        if self.normalization == 'batchnorm':
            for i in range(self.num_layers - 1):
                self.bn_params[i]['mode'] = 'train'

    def eval(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'test'
        if self.normalization == 'batchnorm':
            for i in range(self.num_layers - 1):
                self.bn_params[i]['mode'] = 'test'
