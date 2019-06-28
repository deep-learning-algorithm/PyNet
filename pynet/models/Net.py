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
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass

    @abstractmethod
    def update(self, lr=1e-3, reg=1e-3):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass
