# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:40
# @Author  : zj


import matplotlib.pyplot as plt


class Draw(object):

    def __call__(self, values, title='损失图', xlabel='迭代/次', ylabel='损失值'):
        self.forward(values, title='损失图', xlabel='迭代/次', ylabel='损失值')

    def forward(self, values, title='损失图', xlabel='迭代/次', ylabel='损失值'):
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.plot(values)
        plt.show()
