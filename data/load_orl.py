# -*- coding: utf-8 -*-

# @Time    : 19-6-6 上午9:47
# @Author  : zj


import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

img_path = '../data/mnist/test/0/0.png'

# data_path = '/home/zj/data/att_faces_png/'
data_path = '/home/lab305/Documents/data/att_faces_png/'

dst_size = (23, 28)


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


def load_orl_data(shuffle=True):
    """
    加载ORL数据
    """
    cate_list = []
    data_list = []
    cate_dict = dict()

    num = 0
    file_list = os.listdir(data_path)
    for file_name in file_list:
        file_dir = os.path.join(data_path, file_name)
        img_list = os.listdir(file_dir)
        for img_name in img_list:
            img_path = os.path.join(file_dir, img_name)
            img = read_image(img_path, isGray=True)
            resized_img = resize_image(img, dst_size=dst_size)

            data_list.append(resized_img.reshape(-1))
            # data_list.append(img.reshape(-1))
            cate_list.append(file_name)
            cate_dict[file_name] = num
        num += 1

    x_train, x_test, y_train, y_test = train_test_split(np.array(data_list), np.array(cate_list), train_size=0.8,
                                                        test_size=0.2, shuffle=shuffle)

    y_train_labels = pd.DataFrame(y_train)[0].map(cate_dict).values
    y_test_labels = pd.DataFrame(y_test)[0].map(cate_dict).values

    return x_train, x_test, y_train_labels, y_test_labels
