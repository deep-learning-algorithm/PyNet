# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:40
# @Author  : zj


import matplotlib.pyplot as plt


class Draw(object):

    def __call__(self, values, title='损失图', xlabel='迭代/次', ylabel='损失值'):
        self.forward(values, title='损失图', xlabel='迭代/次', ylabel='损失值')

    def forward(self, values, title='损失图', xlabel='迭代/次', ylabel='损失值'):
        assert isinstance(values, list)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.plot(values)
        plt.show()

    def multi_plot(self, values_list, labels_list, title='损失图', xlabel='迭代/次', ylabel='损失值'):
        assert isinstance(values_list, tuple)
        assert isinstance(labels_list, tuple)
        assert len(values_list) == len(labels_list)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        for i in range(len(values_list)):
            plt.plot(values_list[i], label=labels_list[i])
        plt.legend()
        plt.show()
