# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:30
# @Author  : zj


import numpy as np


class Accuracy(object):

    def __init__(self):
        pass

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        predicted = np.argmax(scores, axis=1)
        return np.mean(predicted == labels), predicted
