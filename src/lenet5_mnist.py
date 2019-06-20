# -*- coding: utf-8 -*-

# @Time    : 19-5-27 下午6:40
# @Author  : zj

import nn
import models
import models.utils as utils
import vision.data
import numpy as np
import time

data_path = '/home/lab305/Documents/data/mnist'

batch_size = 128
momentum = 0.9
learning_rate = 1e-3
reg = 1e-3
epochs = 1000


def lenet5_train():
    x_train, x_test, y_train, y_test = vision.data.load_mnist(data_path, shuffle=True)

    # 标准化
    x_train = x_train / 255.0 - 0.5
    x_test = x_test / 255.0 - 0.5

    net = models.lenet5(momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    accuracy = vision.Accuracy()

    loss_list = []
    train_list = []
    test_list = []
    best_train_accuracy = 0.995
    best_test_accuracy = 0.995

    range_list = np.arange(0, x_train.shape[0] - batch_size, step=batch_size)
    for i in range(epochs):
        total_loss = 0
        num = 0
        start = time.time()
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train[j:j + batch_size]

            scores = net(data)
            loss = criterion(scores, labels)
            total_loss += loss
            num += 1

            grad_out = criterion.backward()
            net.backward(grad_out)
            net.update(lr=learning_rate, reg=reg)
        end = time.time()
        print('one epoch need time: %.3f' % (end - start))
        print('epoch: %d loss: %f' % (i + 1, total_loss / num))
        loss_list.append(total_loss / num)

        if (i % 50) == 50:
            # # 每隔50次降低学习率
            # learning_rate *= 0.5

            train_accuracy = accuracy.compute_v2(x_train, y_train, net, batch_size=batch_size)
            test_accuracy = accuracy.compute_v2(x_test, y_test, net, batch_size=batch_size)
            train_list.append(train_accuracy)
            test_list.append(test_accuracy)

            print(loss_list)
            print(train_list)
            print(test_list)
            if train_accuracy > best_train_accuracy and test_accuracy > best_test_accuracy:
                path = 'lenet5-epochs-%d.pkl' % (i + 1)
                utils.save_params(net.get_params(), path=path)
                break

    draw = vision.Draw()
    draw(loss_list, xlabel='迭代/50次')
    draw.multi_plot((train_list, test_list), ('训练集', '测试集'), title='精度图', xlabel='迭代/50次', ylabel='精度值')


def lenet5_test(model_path):
    params = utils.load_params(model_path)
    print(params)

    net = models.lenet5()
    net.set_params(params)

    x_train, x_test, y_train, y_test = vision.data.load_mnist(data_path, shuffle=True)

    # 标准化
    x_train = x_train / 255.0 - 0.5
    x_test = x_test / 255.0 - 0.5

    accuracy = vision.Accuracy()

    test_accuracy = accuracy.compute_v2(x_test, y_test, net, batch_size=batch_size)
    print(test_accuracy)
    train_accuracy = accuracy.compute_v2(x_train, y_train, net, batch_size=batch_size)
    print(train_accuracy)


if __name__ == '__main__':
    start = time.time()
    lenet5_train()
    end = time.time()
    print('training need time: %.3f' % (end - start))
    # lenet5_test()