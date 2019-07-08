# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:25
# @Author  : zj


import cv2
import pickle


def read_image(img_path, is_gray=False):
    if is_gray:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(img_path)


def resize_image(src, dst_size):
    if src.shape == dst_size:
        return src
    return cv2.resize(src, dst_size)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
        return data_dict
