# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:25
# @Author  : zj


import cv2


def read_image(img_path, is_gray=False):
    if is_gray:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(img_path)


def resize_image(src, dst_size):
    if src.shape == dst_size:
        return src
    return cv2.resize(src, dst_size)


def change_channel(inputs):
    if len(inputs.shape) == 2:
        # 灰度图
        dst_shape = [1]
        dst_shape.extend(inputs.shape)
        return inputs.reshape(dst_shape)
    else:
        # 彩色图
        return inputs.transpose(2, 0, 1)
