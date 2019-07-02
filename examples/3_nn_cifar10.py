# -*- coding: utf-8 -*-

# @Time    : 19-7-1 下午6:53
# @Author  : zj

import numpy as np
import pynet.models as models
import pynet
import pynet.optim as optim
from pynet.vision.data import cifar
import pynet.nn as nn
import plt

data_path = '/home/zj/data/decompress_cifar_10'
# data_path = '/home/lab305/Documents/zj/data/decompress_cifar_10'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = cifar.load_cifar10(data_path, shuffle=True, is_flatten=True)

    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_test,
        'y_val': y_test
    }

    model = models.ThreeLayerNet(num_in=3072, num_h1=2000, num_h2=800, num_out=10, dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.params, lr=1e-3)

    solver = pynet.Solver(model, data, criterion, optimizer, batch_size=256, num_epochs=50,
                          reg=1e-3, print_every=1)
    solver.train()

    plt.draw_loss(solver.loss_history)
    plt.draw_acc((solver.train_acc_history, solver.val_acc_history), ('train', 'val'))
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))
