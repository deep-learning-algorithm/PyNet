# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:37
# @Author  : zj

from .utils import *
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

dst_size = (23, 28)


def load_orl(orl_path, shuffle=True):
    """
    加载ORL数据
    """
    cate_list = []
    data_list = []
    cate_dict = dict()

    num = 0
    file_list = os.listdir(orl_path)
    for file_name in file_list:
        file_dir = os.path.join(orl_path, file_name)
        img_list = os.listdir(file_dir)
        for img_name in img_list:
            img_path = os.path.join(file_dir, img_name)
            img = read_image(img_path, is_gray=True)
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
