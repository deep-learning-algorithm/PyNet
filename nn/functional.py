# -*- coding: utf-8 -*-

# @Time    : 19-6-7 下午2:46
# @Author  : zj

import numpy as np


def dropout(shape, p):
    assert len(shape) == 2
    res = (np.random.ranf(shape) < p) / p

    if np.sum(res) == 0:
        return 1.0 / p
    return res


def dropout2d(shape, p):
    assert len(shape) == 4
    N, C, H, W = shape[:4]
    U = (np.random.rand(N * C, 1) < p) / p
    res = np.ones((N * C, H * W))
    res *= U

    if np.sum(res) == 0:
        return 1.0 / p
    return res.reshape(N, C, H, W)


if __name__ == '__main__':
    # res = dropout((3, 4), 0.5)
    res = dropout2d((2, 3, 2, 2), 0.5)
    print(res)
