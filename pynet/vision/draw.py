# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:40
# @Author  : zj


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Draw(object):

    def __call__(self, values, title='损失图', xlabel='迭代/次', ylabel='损失值'):
        self.forward(values, title='损失图', xlabel='迭代/次', ylabel='损失值')

    def forward(self, values, title='损失图', xlabel='迭代/次', ylabel='损失值', save_path='./loss.png'):
        assert isinstance(values, list)
        f = plt.figure()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.plot(values)
        plt.savefig(save_path)
        plt.show()

    def multi_plot(self, values_list, labels_list, title='损失图', xlabel='迭代/次', ylabel='损失值', save_path='./loss.png'):
        assert isinstance(values_list, tuple)
        assert isinstance(labels_list, tuple)
        assert len(values_list) == len(labels_list)

        f = plt.figure()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        for i in range(len(values_list)):
            plt.plot(values_list[i], label=labels_list[i])
        plt.legend()
        plt.savefig(save_path)
        plt.show()
