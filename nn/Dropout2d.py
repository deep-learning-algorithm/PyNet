# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:37
# @Author  : zj

import numpy as np


class Dropout2d(object):

    def __call__(self, shape, p):
        return self.forward(shape, p)

    def forward(self, shape, p):
        assert len(shape) == 4
        N, C, H, W = shape[:4]
        U = (np.random.rand(N * C, 1) < p) / p
        res = np.ones((N * C, H * W))
        res *= U

        if np.sum(res) == 0:
            return 1.0 / p
        return res.reshape(N, C, H, W)
