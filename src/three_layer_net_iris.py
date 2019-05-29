# -*- coding: utf-8 -*-

# @Time    : 19-5-29 上午10:08
# @Author  : zj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.model_selection import train_test_split
from nn.nets import *
from nn.net_utils import *

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

iris_path = '/home/zj/data/iris-species/Iris.csv'


def load_data(shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(iris_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    return x_train, x_test, y_train, y_test


def compute_accuracy(score, y):
    predicted = np.argmax(score, axis=1)
    return np.mean(predicted == y)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data(shuffle=True, tsize=0.8)

    net = ThreeLayerNet(D, H1, H2, K)
    criterion = CrossEntropyLoss()

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

            train_accuracy = compute_accuracy(scores, y_train)
            test_accuracy = compute_accuracy(net(x_test), y_test)
            train_list.append(train_accuracy)
            test_list.append(test_accuracy)
            if train_accuracy >= 0.9999 and test_accuracy >= 0.9999:
                save_params(net.get_params(), path='three_layer_net_iris.pkl')
                break
        if (i % 10000) == 9999:
            # 每隔10000次降低学习率
            learning_rate *= 0.5
    print(loss_list)
    print(train_list)
    print(test_list)

    plt.figure(1)
    plt.title('损失图')
    plt.ylabel('损失值')
    plt.xlabel('迭代/500次')
    plt.plot(loss_list)

    plt.figure(2)
    plt.title('精度图')
    plt.ylabel('精度值')
    plt.xlabel('迭代/500次')
    plt.plot(train_list, label='训练集')
    plt.plot(test_list, label='测试集')
    plt.legend()
    plt.show()
