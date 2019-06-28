# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午4:18
# @Author  : zj


import numpy as np


def load_xor():
    xor_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_labels = np.array([0, 1, 1, 0])

    return xor_data, xor_labels
