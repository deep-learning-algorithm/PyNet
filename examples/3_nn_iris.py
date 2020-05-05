# -*- coding: utf-8 -*-

# @Time    : 19-7-1 下午4:39
# @Author  : zj


import numpy as np
import pynet
import pynet.optim as optim
import pynet.models as models
import pynet.nn as nn
from pynet.vision.data import iris
from pynet.vision import Draw

data_path = '/home/zj/data/iris-species/Iris.csv'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = iris.load_iris(data_path, shuffle=True, tsize=0.8)

    mean = np.mean(x_train, axis=0, keepdims=True)
    x_train -= mean
    x_test -= mean

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_test,
        'y_val': y_test
    }

    model = models.ThreeLayerNet(num_in=4, num_h1=40, num_h2=20, num_out=3, dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.params, lr=1e-3, momentum=0.5, nesterov=True)

    solver = pynet.Solver(model, data, criterion, optimizer, batch_size=120, num_epochs=50000,
                          reg=1e-3, print_every=500)
    solver.train()

    plt = Draw()
    plt(solver.loss_history)
    plt.multi_plot((solver.train_acc_history, solver.val_acc_history), ('train', 'val'),
                   title='准确率', xlabel='迭代/次', ylabel='准确率')
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))
