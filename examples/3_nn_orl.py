# -*- coding: utf-8 -*-

# @Time    : 19-7-1 下午6:40
# @Author  : zj

import numpy as np
from pynet.models import ThreeLayerNet
from pynet import Solver
from pynet.vision.data import orl
import pynet.nn as nn
import plt

data_path = '/home/zj/data/att_faces_png'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = orl.load_orl(data_path, shuffle=True)

    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_test,
        'y_val': y_test
    }

    model = ThreeLayerNet(num_in=644, num_h1=2000, num_h2=800, num_out=40)
    criterion = nn.CrossEntropyLoss()

    solver = Solver(model, data, criterion, update_rule='sgd', batch_size=4, num_epochs=100,
                    reg=1e-3, print_every=1, optim_config={'learning_rate': 2e-2})
    solver.train()

    plt.draw_loss(solver.loss_history)
    plt.draw_acc((solver.train_acc_history, solver.val_acc_history), ('train', 'val'))
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))