# -*- coding: utf-8 -*-

# @Time    : 19-7-2 下午4:30
# @Author  : zj


from abc import ABCMeta, abstractmethod

__all__ = ['Optimizer']


class Optimizer(metaclass=ABCMeta):

    def __init__(self, params, defaults):
        self.params = params

        self.optim_configs = {}
        for p in params.keys():
            # d = {k: v for k, v in defaults.items()}
            d = defaults.copy()
            self.optim_configs[p] = d

    @abstractmethod
    def step(self, grad):
        pass
