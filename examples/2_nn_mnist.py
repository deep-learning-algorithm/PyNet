# -*- coding: utf-8 -*-

# @Time    : 19-6-30 下午3:13
# @Author  : zj

import pynet
import pynet.models as models
import pynet.optim as optim
import pynet.nn as nn
from pynet.vision.data import mnist
from pynet.vision import Draw

data_path = '~/data/decompress_mnist'

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

    model = models.TwoLayerNet(num_in=784, num_hidden=200, num_out=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.params)

    solver = pynet.Solver(model, data, criterion, optimizer, batch_size=128, num_epochs=10)
    solver.train()

    plt = Draw()
    plt(solver.loss_history)
    plt.multi_plot((solver.train_acc_history, solver.val_acc_history), ('train', 'val'),
                   title='准确率', xlabel='迭代/次', ylabel='准确率', save_path='acc.png')
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))
