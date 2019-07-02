# -*- coding: utf-8 -*-

# @Time    : 19-7-2 下午4:27
# @Author  : zj

from .LRScheduler import LRScheduler


class StepLR(LRScheduler):

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma

        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size) for base_lr in self.base_lrs]
