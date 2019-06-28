# -*- coding: utf-8 -*-

# @Time    : 19-5-26 下午1:45
# @Author  : zj

from builtins import range
import numpy as np
import time


def get_pool2row_indices(x_shape, field_height, field_width, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H - field_height) % stride == 0
    assert (W - field_height) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    # 行坐标
    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(field_height), field_width)

    # 列坐标
    j0 = stride * np.tile(np.arange(out_width), out_height * C)
    j1 = np.tile(np.arange(field_width), field_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), out_height * out_width).reshape(-1, 1)

    return (k, i, j)


def pool2row_indices(x, field_height, field_width, stride=1):
    k, i, j = get_pool2row_indices(x.shape, field_height, field_width, stride)
    rows = x.copy()[:, k, i, j]

    return rows.reshape(-1, field_height * field_width)


def row2pool_indices(rows, x_shape, field_height=2, field_width=2, stride=2, isstinct=False):
    N, C, H, W = x_shape
    x = np.zeros(x_shape, dtype=rows.dtype)
    k, i, j = get_pool2row_indices(x_shape, field_height, field_width, stride)
    rows_reshaped = rows.reshape(N, -1, field_height * field_width)
    np.add.at(x, (slice(None), k, i, j), rows_reshaped)

    if isstinct and (stride < field_height or stride < field_width):
        x_ones = np.ones(x.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        return x / x_zeros

    return x


if __name__ == '__main__':
    # x = np.arange(32).reshape(2, 1, 4, 4)
    # print(x)
    x = np.ones((128, 3, 32, 32))
    rows = pool2row_indices(x, 4, 4, stride=2)
    # print(rows)
    print(rows.shape)
    pool = row2pool_indices(rows, x.shape, field_height=4, field_width=4, stride=2, isstinct=True)
    # print(pool)
    print(np.sum(x == pool))
