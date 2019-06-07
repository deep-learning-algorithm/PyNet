# -*- coding: utf-8 -*-

# @Time    : 19-5-27 下午6:40
# @Author  : zj

from nn.nets import *
from nn.net_utils import *
from data.load_mnist import *
import numpy as np
import matplotlib.pyplot as plt
import time

model_path = '../model/lenet5-epochs-250.pkl'

batch_size = 128
momentum = 0.9


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

    net = LeNet5(momentum=momentum)
    criterion = CrossEntropyLoss()

    learning_rate = 1e-3
    reg = 1e-3
    epochs = 1000

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

            train_accuracy = compute_accuracy(x_train, y_train, net, batch_size=batch_size)
            test_accuracy = compute_accuracy(x_test, y_test, net, batch_size=batch_size)
            train_list.append(train_accuracy)
            test_list.append(test_accuracy)

            print(loss_list)
            print(train_list)
            print(test_list)
            if train_accuracy > best_train_accuracy and test_accuracy > best_test_accuracy:
                best_train_accuracy = train_accuracy
                best_test_accuracy = test_accuracy

                path = 'lenet5-epochs-%d.pkl' % (i + 1)
                save_params(net.get_params(), path=path)
                break

    plt.figure(1)
    plt.title('损失图')
    plt.ylabel('损失值')
    plt.xlabel('迭代/50次')
    plt.plot(loss_list)

    plt.figure(2)
    plt.title('精度图')
    plt.ylabel('精度值')
    plt.xlabel('迭代/50次')
    plt.plot(train_list, label='训练集')
    plt.plot(test_list, label='测试集')
    plt.legend()
    plt.show()


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
    start = time.time()
    lenet5_train()
    end = time.time()
    print('training need time: %.3f' % (end - start))
    # lenet5_test()
