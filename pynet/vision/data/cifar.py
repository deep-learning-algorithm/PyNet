# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:22
# @Author  : zj

import numpy as np
import os
from sklearn.model_selection import train_test_split
from .utils import *

train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

test_batch = 'test_batch'


def load_CIFAR_batch(file_path):
    """ load single batch of cifar """
    data_dict = load_pickle(file_path)
    data = data_dict['data']
    labels = data_dict['labels']

    data = data.reshape(10000, 3, 32, 32).astype("float")
    labels = np.array(labels)
    return data, labels


def load_CIFAR10(file_dir):
    """ load all of cifar """
    xs = []
    ys = []
    for filename in train_list:
        file_path = os.path.join(file_dir, filename)
        data, labels = load_CIFAR_batch(file_path)
        xs.append(data)
        ys.append(labels)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    x_test, y_test = load_CIFAR_batch(os.path.join(file_dir, test_batch))

    return x_train, y_train, x_test, y_test


def get_CIFAR10_data(cifar_dir, val_size=0.05, normalize=True):
    """
    加载CIFAR10数据，从训练集中分类验证集数据
    :param cifar_dir: cifar解压文件路径
    :param val_size: 浮点数，表示验证集占整个训练集的百分比
    :param normalize: 是否初始化为零均值，1方差
    :return: dict，保存训练集、验证集以及测试集的数据和标签
    """
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar_dir)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=False)

    # Normalize the data: subtract the mean image and divide the variance
    if normalize:
        x_train = x_train / 255 - 0.5
        x_val = x_val / 255 - 0.5
        x_test = x_test / 255 - 0.5

    # Package data into a dictionary
    return {
        'X_train': x_train, 'y_train': y_train,
        'X_val': x_val, 'y_val': y_val,
        'X_test': x_test, 'y_test': y_test,
    }
