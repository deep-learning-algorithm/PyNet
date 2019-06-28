# -*- coding: utf-8 -*-

# @Time    : 19-5-25 下午4:17
# @Author  : zj

from builtins import range
import numpy as np
import time


def get_im2row_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    # 行坐标
    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i1 = np.repeat(np.arange(field_height), field_width)
    i1 = np.tile(i1, C)

    # 列坐标
    j0 = stride * np.tile(np.arange(out_width), out_height)
    j1 = np.tile(np.arange(field_width), field_height * C)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(1, -1)

    return (k, i, j)


def im2row_indices(x, field_height, field_width, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2row_indices(x.shape, field_height, field_width, padding, stride)

    rows = x_padded[:, k, i, j]
    C = x.shape[1]
    # 逐图像采集
    rows = rows.reshape(-1, field_height * field_width * C)
    return rows


def row2im_indices(rows, x_shape, field_height=3, field_width=3, padding=1, stride=1, isstinct=False):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=rows.dtype)
    k, i, j = get_im2row_indices(x_shape, field_height, field_width, padding,
                                 stride)
    rows_reshaped = rows.reshape(N, -1, C * field_height * field_width)
    np.add.at(x_padded, (slice(None), k, i, j), rows_reshaped)

    if isstinct:
        # 计算叠加倍数，恢复原图
        x_ones = np.ones(x_padded.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x_padded.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        x_padded = x_padded / x_zeros

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]
