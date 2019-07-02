# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:50
# @Author  : zj

from .utils import *
import os
import numpy as np


def load_mnist(mnist_path, dst_size=None, shuffle=True, is_flatten=False):
    """
    加载mnist数据
    """
    train_dir = os.path.join(mnist_path, 'train')
    test_dir = os.path.join(mnist_path, 'test')

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_file_list = []
    cate_list = list(range(10))
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
            img = read_image(file_path, is_gray=True)
            if img is not None:
                if dst_size is not None:
                    img = resize_image(img, dst_size=dst_size)
                if is_flatten:
                    x_test.append(img.reshape(-1))
                else:
                    h, w = img.shape[:2]
                    x_test.append(img.reshape(1, h, w))
                y_test.append(i)

    train_file_list = np.array(train_file_list)
    if shuffle:
        np.random.shuffle(train_file_list)

    # 读取训练集图像
    for file_path in train_file_list:
        img = read_image(file_path, is_gray=True)
        if img is not None:
            if dst_size is not None:
                img = resize_image(img, dst_size=dst_size)

            if is_flatten:
                x_train.append(img.reshape(-1))
            else:
                h, w = img.shape[:2]
                x_train.append(img.reshape(1, h, w))
            y_train.append(int(os.path.split(file_path)[0].split('/')[-1]))

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
