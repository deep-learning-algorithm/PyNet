# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:56
# @Author  : zj

import numpy as np


class Dropout(object):

    def __call__(self, inputs, dropout_param):
        return self.forward(inputs, dropout_param)

    def forward(self, inputs, dropout_param):
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            np.random.seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            mask = (np.random.ranf(inputs.shape) < p) / p
            out = inputs * mask
        elif mode == 'test':
            out = inputs

        cache = (dropout_param, mask)
        #     print(mask.shape)
        out = out.astype(inputs.dtype, copy=False)

        return out, cache

    def backward(self, dout, cache):
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout

        return dx
