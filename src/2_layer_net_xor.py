# -*- coding: utf-8 -*-

# @Time    : 19-5-27 下午6:39
# @Author  : zj

import models
import vision
import vision.data
import nn
from data.load_mnist import *

"""
XOR问题
"""


def two_layer_train():
    net = models.two_layer_net(num_in=2, num_hidden=6, num_out=2)

    input_array, xor_array = vision.data.load_xor()

    criterion = nn.CrossEntropyLoss()

    lr = 1e-1
    reg = 1e-3

    loss_list = []
    total_loss = 0
    for i in range(10000):
        scores = net.forward(input_array)
        total_loss += criterion.forward(scores, xor_array)
        grad_out = criterion.backward()
        net.backward(grad_out)
        net.update(lr=lr, reg=reg)

        if (i % 500) == 499:
            print('epoch: %d loss: %.4f' % (i + 1, total_loss / 500))
            loss_list.append(total_loss / 500)
            total_loss = 0
    draw = vision.Draw()
    draw(loss_list, '逻辑异或')

    params = net.get_params()
    print('FC1: {}'.format(params['fc1']))
    print('FC2: {}'.format(params['fc2']))
    # save_params(params, path='./two_layer_net.pkl')

    scores = net.forward(input_array)
    accuracy = vision.Accuracy()
    res, predict = accuracy(scores, xor_array)
    print('labels: ' + str(xor_array))
    print('predict: ' + str(predict))
    print('training accuracy: %.2f %%' % (res * 100))


def test():
    # params = load_params('./two_layer_net.pkl')
    # print(params)
    net = models.two_layer_net(num_in=2, num_hidden=6, num_out=2)
    # net.set_params(params)

    input_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_array = np.array([0, 1, 1, 0])

    for item in input_array:
        print(net.forward(np.atleast_2d(item)))


if __name__ == '__main__':
    two_layer_train()
    # test()
