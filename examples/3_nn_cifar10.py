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

data_path = '/home/lab305/Documents/zj/data/cifar_10/cifar-10-batches-py'

if __name__ == '__main__':
    data_dict = cifar.get_CIFAR10_data(data_path)

    model = models.FCNet([2000, 800], input_dim=3072, num_classes=10, normalization='batchnorm')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.params, lr=1e-3)

    solver = pynet.Solver(model, data_dict, criterion, optimizer, batch_size=256, num_epochs=250,
                          reg=1e-3, print_every=1)
    solver.train()

    plt.draw_loss(solver.loss_history)
    plt.draw_acc((solver.train_acc_history, solver.val_acc_history), ('train', 'val'))
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))
