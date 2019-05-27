# -*- coding: utf-8 -*-

# @Time    : 19-5-26 下午4:05
# @Author  : zj

import numpy as np
import os
import cv2

img_path = '../data/mnist/test/0/0.png'

data_path = '../data/mnist/'

cate_list = list(range(10))

dst_size = (32, 32)


def read_image(img_path, isGray=False):
    if isGray:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(img_path)


def resize_image(src, dst_size):
    return cv2.resize(src, dst_size)


def change_channel(input):
    if len(input.shape) == 2:
        # 灰度图
        dst_shape = [1]
        dst_shape.extend(input.shape)
        return input.reshape(dst_shape)
    else:
        # 彩色图
        return input.transpose(2, 0, 1)


def load_mnist_data(shuffle=True):
    """
    加载mnist数据
    """
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

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
            img = read_image(file_path, isGray=True)
            if img is not None:
                x_test.append(change_channel(resize_image(img, dst_size=dst_size)))
                y_test.append(i)

    train_file_list = np.array(train_file_list)
    if shuffle:
        np.random.shuffle(train_file_list)

    # 读取训练集图像
    for file_path in train_file_list:
        img = read_image(file_path, isGray=True)
        if img is not None:
            x_train.append(change_channel(resize_image(img, dst_size=dst_size)))
            y_train.append(int(os.path.split(file_path)[0].split('/')[-1]))

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
