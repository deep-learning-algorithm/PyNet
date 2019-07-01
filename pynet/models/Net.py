# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:06
# @Author  : zj

from abc import ABCMeta, abstractmethod

__all__ = ['Net']


class Net(metaclass=ABCMeta):
    """
    所有神经网络模型基类
    """

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass
