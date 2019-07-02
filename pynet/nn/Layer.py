# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:24
# @Author  : zj

from abc import ABCMeta, abstractmethod

__all__ = ['Layer']


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out, cache):
        pass
