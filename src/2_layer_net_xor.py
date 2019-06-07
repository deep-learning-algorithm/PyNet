# -*- coding: utf-8 -*-

# @Time    : 19-5-27 下午6:39
# @Author  : zj

from nn.nets import *
from nn.net_utils import *
from data.load_mnist import *
import matplotlib.pyplot as plt

"""
XOR问题
"""

def compute_accuracy(scores, labels):
    predicted = np.argmax(scores, axis=1)
    return np.mean(predicted == labels), predicted


def draw(loss_list, title='损失图'):
    plt.title(title)
    plt.ylabel('损失值')
    plt.xlabel('迭代/500次')
    plt.plot(loss_list)
    plt.show()


def two_layer_train():
    net = TwoLayerNet(2, 6, 2)

    input_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_array = np.array([0, 1, 1, 0])

    criterion = CrossEntropyLoss()

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
    draw(loss_list, '逻辑异或')

    params = net.get_params()
    print('FC1: {}'.format(params['fc1']))
    print('FC2: {}'.format(params['fc2']))
    save_params(params, path='./two_layer_net.pkl')

    scores = net.forward(input_array)
    res, predict = compute_accuracy(scores, xor_array)
    print('labels: ' + str(xor_array))
    print('predict: ' + str(predict))
    print('training accuracy: %.2f %%' % (res * 100))


def test():
    params = load_params('./two_layer_net.pkl')
    # print(params)
    net = TwoLayerNet(2, 6, 2)
    net.set_params(params)

    input_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_array = np.array([0, 1, 1, 0])

    for item in input_array:
        print(net.forward(np.atleast_2d(item)))


if __name__ == '__main__':
    # two_layer_train()
    test()
