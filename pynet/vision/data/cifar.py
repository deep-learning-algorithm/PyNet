# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:22
# @Author  : zj

import numpy as np
import os
from .utils import *

data_path = '/home/lab305/Documents/data/decompress_cifar_10'

cate_list = list(range(10))

dst_size = (32, 32)


def load_cifar10(cifar10_path, shuffle=True, is_flatten=False):
    """
    加载cifar10
    """

    train_dir = os.path.join(cifar10_path, 'train')
    test_dir = os.path.join(cifar10_path, 'test')

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_file_list = []
    for i in cate_list:
        data_dir = os.path.join(train_dir, str(i))
        file_list = os.listdir(data_dir)
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            train_file_list.append(file_path)

        # 读取测试集图像
        data_dir = os.path.join(test_dir, str(i))
        file_list = os.listdir(data_dir)
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            img = read_image(file_path)
            if img is not None:
                y_test.append(i)
                if is_flatten:
                    x_test.append(img.reshape(-1))
                else:
                    x_test.append(np.transpose(img, (2, 0, 1)))

    train_file_list = np.array(train_file_list)
    if shuffle:
        np.random.shuffle(train_file_list)

    # 读取训练集图像
    for file_path in train_file_list:
        img = read_image(file_path)
        if img is not None:
            y_train.append(int(os.path.split(file_path)[0].split('/')[-1]))
            if is_flatten:
                x_train.append(img.reshape(-1))
            else:
                x_train.append(np.transpose(img, (2, 0, 1)))

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
