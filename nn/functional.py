# -*- coding: utf-8 -*-

# @Time    : 19-6-7 下午2:46
# @Author  : zj

import numpy as np


def dropout(shape, p):
    assert len(shape) == 2
    return (np.random.ranf(shape) < p) / p


def dropout2d(shape, p):
    assert len(shape) == 4
    N, C, H, W = shape[:4]
    U = (np.random.rand(N, C) < p) / p
    res = np.ones(shape)
    for i in range(N):
        for j in range(C):
            res[i, j] *= U[i, j]

    return res


if __name__ == '__main__':
    res = dropout((3, 4), 0.5)
    # res = dropout2d((1, 4, 2, 2), 0.5)
    print(res)
