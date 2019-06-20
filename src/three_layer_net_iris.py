# -*- coding: utf-8 -*-

# @Time    : 19-5-29 上午10:08
# @Author  : zj

import os
import numpy as np
import vision.data
import models
import models.utils as utils
import nn

"""
iris分类
"""

# 批量大小
N = 120
# 输入维数
D = 4
# 隐藏层大小
H1 = 20
H2 = 20
# 输出类别
K = 3

# 迭代次数
epochs = 50000

# 学习率
learning_rate = 1e-3
# 正则化强度
lambda_rate = 1e-3

# iris_path = '/home/zj/data/iris-species/Iris.csv'
iris_path = '/home/lab305/Documents/data/iris-species/Iris.csv'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = vision.data.load_iris(iris_path, shuffle=True, tsize=0.8)

    net = models.three_layer_net(num_in=D, num_h1=H1, num_h2=H2, num_out=K)
    criterion = nn.CrossEntropyLoss()

    accuracy = vision.Accuracy()

    loss_list = []
    train_list = []
    test_list = []
    total_loss = 0
    for i in range(epochs):
        scores = net(x_train)
        total_loss += criterion(scores, y_train)

        grad_out = criterion.backward()
        net.backward(grad_out)
        net.update(lr=learning_rate, reg=0)

        if (i % 500) == 499:
            print('epoch: %d loss: %f' % (i + 1, total_loss / 500))
            loss_list.append(total_loss / 500)
            total_loss = 0

            train_accuracy, res = accuracy(scores, y_train)
            test_accuracy, res = accuracy(net(x_test), y_test)
            train_list.append(train_accuracy)
            test_list.append(test_accuracy)
            if train_accuracy >= 0.9999 and test_accuracy >= 0.9999:
                utils.save_params(net.get_params(), path=os.path.join(os.getcwd(), 'three_layer_net_iris.pkl'))
                break
        if (i % 10000) == 9999:
            # 每隔10000次降低学习率
            learning_rate *= 0.5
    print(loss_list)
    print(train_list)
    print(test_list)

    draw = vision.Draw()

    draw(loss_list, xlabel='迭代/500次')
    draw.multi_plot((train_list, test_list), ('训练集', '测试集'), title='精度图', xlabel='迭代/500次', ylabel='精度值')
