# -*- coding: utf-8 -*-

# @Time    : 19-6-30 下午3:13
# @Author  : zj

from pynet.models import TwoLayerNet
from pynet import Solver
from pynet.vision.data import mnist
import pynet.nn as nn
import plt

data_path = '/home/zj/data/decompress_mnist'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = mnist.load_mnist(data_path, shuffle=True, is_flatten=True)

    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_test,
        'y_val': y_test
    }

    model = TwoLayerNet(num_in=784, num_hidden=200, num_out=10)
    criterion = nn.CrossEntropyLoss()

    solver = Solver(model, data, criterion, update_rule='sgd', batch_size=128, num_epochs=10, reg=1e-3)
    solver.train()

    plt.draw_loss(solver.loss_history)
    plt.draw_acc((solver.train_acc_history, solver.val_acc_history), ('train', 'val'))
