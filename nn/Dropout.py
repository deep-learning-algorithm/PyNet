# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:56
# @Author  : zj

import numpy as np


class Dropout(object):

    def __call__(self, shape, p):
        return self.forward(shape, p)

    def forward(self, shape, p):
        assert len(shape) == 2
        res = (np.random.ranf(shape) < p) / p

        if np.sum(res) == 0:
            return 1.0 / p
        return res
