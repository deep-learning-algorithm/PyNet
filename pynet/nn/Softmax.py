# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午8:07
# @Author  : zj

import numpy as np

__all__ = ['Softmax']


class Softmax(object):
    """
    softmax评分
    """

    def __init__(self):
        pass

    def __call__(self, scores):
        return self.forward(scores)

    def forward(self, scores):
        # scores.shape == [N, C]
        assert len(scores.shape) == 2
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        probs = expscores / np.sum(expscores, axis=1, keepdims=True)

        return probs
