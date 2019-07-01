# -*- coding: utf-8 -*-

# @Time    : 19-7-1 下午4:39
# @Author  : zj


import numpy as np
from pynet.models import ThreeLayerNet
from pynet import Solver
from pynet.vision.data import iris
import pynet.nn as nn
import plt

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

    model = ThreeLayerNet(num_in=4, num_h1=40, num_h2=20, num_out=3)
    criterion = nn.CrossEntropyLoss()

    solver = Solver(model, data, criterion, update_rule='sgd', batch_size=120, num_epochs=10000,
                    reg=1e-3, print_every=100, optim_config={'learning_rate': 1e-3})
    solver.train()

    plt.draw_loss(solver.loss_history)
    plt.draw_acc((solver.train_acc_history, solver.val_acc_history), ('train', 'val'))
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))
