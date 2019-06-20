# -*- coding: utf-8 -*-

# @Time    : 19-6-6 上午9:47
# @Author  : zj

import numpy as np
import time

import vision.data
import models
import models.utils as utils
import nn

# 批量大小
batch_size = 4
# 输入维数
D = 644
# 隐藏层大小
H1 = 2000
H2 = 800
# 输出类别
K = 40

# 学习率
lr = 2e-2
# 正则化强度
reg_rate = 1e-3

data_path = '/home/lab305/Documents/data/att_faces_png'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = vision.data.load_orl(data_path, shuffle=True)

    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5

    net = models.three_layer_net(num_in=D, num_h1=H1, num_h2=H2, num_out=K, p_h=0.5)
    criterion = nn.CrossEntropyLoss()

    accuracy = vision.Accuracy()

    loss_list = []
    train_accuracy_list = []
    best_train_accuracy = 0.995
    best_test_accuracy = 0

    range_list = np.arange(0, x_train.shape[0] - batch_size, step=batch_size)
    for i in range(200):
        start = time.time()
        total_loss = 0
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train[j:j + batch_size]

            scores = net.forward(data)
            total_loss += criterion.forward(scores, labels)
            dout = criterion.backward()
            net.backward(dout)
            net.update(lr=lr, reg=reg_rate)
        end = time.time()

        avg_loss = total_loss / len(range_list)
        loss_list.append(float('%.4f' % avg_loss))
        print('epoch: %d time: %f loss: %.4f' % (i + 1, end - start, avg_loss))

        # 计算训练数据集检测精度
        train_accuracy = accuracy.compute_v2(x_train, y_train, net, batch_size=batch_size)
        train_accuracy_list.append(float('%.4f' % train_accuracy))
        if best_train_accuracy < train_accuracy:
            best_train_accuracy = train_accuracy

            test_accuracy = accuracy.compute_v2(x_test, y_test, net, batch_size=batch_size)
            if best_test_accuracy < test_accuracy:
                best_test_accuracy = test_accuracy
                utils.save_params(net.get_params(), path='three-dropout-nn-epochs-%d.pkl' % (i + 1))

        print('best train accuracy: %.2f %%   best test accuracy: %.2f %%' % (
            best_train_accuracy * 100, best_test_accuracy * 100))
        print(loss_list)
        print(train_accuracy_list)

        if i % 50 == 49:
            lr /= 2

    draw = vision.Draw()
    draw(loss_list, title='mnist', xlabel='迭代/次')
    draw(train_accuracy_list, title='训练', ylabel='检测精度', xlabel='迭代/次')
