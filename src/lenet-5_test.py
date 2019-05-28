# -*- coding: utf-8 -*-

# @Time    : 19-5-27 下午6:40
# @Author  : zj

from nn.nets import *
from nn.net_utils import *
from src.load_mnist import *
import numpy as np
import matplotlib.pyplot as plt
import time

lr = 1e-3
reg = 1e-3
batch_size = 128
epochs = 1000

model_path = '../model/lenet5-epochs-250.pkl'


def compute_accuracy(x_test, y_test, net, batch_size=128):
    total_accuracy = 0
    num = 0
    range_list = np.arange(0, x_test.shape[0] - batch_size, step=batch_size)
    for i in range_list:
        data = x_test[i:i + batch_size]
        labels = y_test[i:i + batch_size]

        scores = net.forward(data)
        predicted = np.argmax(scores, axis=1)
        total_accuracy += np.mean(predicted == labels)
        num += 1
    return total_accuracy / num


def draw(loss_list, title='损失图'):
    plt.title(title)
    plt.ylabel('损失值')
    plt.xlabel('迭代/500次')
    plt.plot(loss_list)
    plt.show()


def lenet5_train():
    x_train, x_test, y_train, y_test = load_mnist_data(shuffle=True)

    # 标准化
    x_train = x_train / 255.0 - 0.5
    x_test = x_test / 255.0 - 0.5

    net = LeNet5()
    criterion = CrossEntropyLoss()

    loss_list = []
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
            net.update(lr=lr, reg=reg)
        end = time.time()
        print('one epoch need time: %.3f' % (end - start))
        print('epoch: %d loss: %f' % (i + 1, total_loss / num))
        loss_list.append(total_loss / num)
        # draw(loss_list)
        if i % 50 == 49:
            path = 'lenet5-epochs-%d.pkl' % (i + 1)
            params = net.get_params()
            save_params(params, path=path)
            test_accuracy = compute_accuracy(x_test, y_test, net, batch_size=batch_size)
            print('epochs: %d test_accuracy: %f' % (i + 1, test_accuracy))
            print('loss: %s' % loss_list)


def lenet5_test():
    params = load_params(model_path)
    print(params)

    net = LeNet5()
    net.set_params(params)

    x_train, x_test, y_train, y_test = load_mnist_data(shuffle=True)

    # 标准化
    x_train = x_train / 255.0 - 0.5
    x_test = x_test / 255.0 - 0.5

    test_accuracy = compute_accuracy(x_test, y_test, net, batch_size=batch_size)
    print(test_accuracy)
    train_accuracy = compute_accuracy(x_train, y_train, net, batch_size=batch_size)
    print(train_accuracy)


if __name__ == '__main__':
    # start = time.time()
    # lenet5_train()
    # end = time.time()
    # print('training need time: %.3f' % (end - start))
    lenet5_test()
