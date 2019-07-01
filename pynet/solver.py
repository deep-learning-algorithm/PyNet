# -*- coding: utf-8 -*-

# @Time    : 19-6-29 下午8:33
# @Author  : zj

import pynet.optim as optim
import numpy as np
import time

__all__ = ['Solver']


class Solver(object):

    def __init__(self, model, data, criterion, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.criterion = criterion

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 8)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.reg = kwargs.pop('reg', 1e-3)
        self.use_reg = self.reg != 0

        self.print_every = kwargs.pop('print_every', 1)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('未识别参数: %s' % extra)

        if not hasattr(optim, self.update_rule):
            raise ValueError('无效的更新规则： %s' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        self.current_epoch = 0
        self.best_train_acc = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.optim_configs = {}
        for p in self.model.params.keys():
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self, X_batch, y_batch):
        scores = self.model.forward(X_batch)
        loss, probs = self.criterion.forward(scores, y_batch)
        if self.use_reg:
            for k in self.model.params.keys():
                if 'W' in k:
                    loss += 0.5 * self.reg * np.sum(self.model.params[k] ** 2)

        grad_out = self.criterion.backward(probs, y_batch)
        grad = self.model.backward(grad_out)
        if self.use_reg:
            for k in grad.keys():
                if 'W' in k:
                    grad[k] += self.reg * self.model.params[k]

        for p, w in self.model.params.items():
            dw = grad[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

        return loss

    def check_accuracy(self, X, y, num_samples=None, batch_size=8):
        """
        精度测试，如果num_samples小于X长度，则从X中采样num_samples个图片进行检测
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        if N < batch_size:
            batch_size = N
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.forward(X[start:end])
            y_pred.extend(np.argmax(scores, axis=1))
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)

        for i in range(self.num_epochs):
            self.current_epoch = i + 1
            start = time.time()
            total_loss = 0.
            self.model.train()
            for j in range(iterations_per_epoch):
                idx_start = j * self.batch_size
                idx_end = (j + 1) * self.batch_size
                X_batch = self.X_train[idx_start:idx_end]
                y_batch = self.y_train[idx_start:idx_end]

                total_loss += self._step(X_batch, y_batch)
            end = time.time()

            if self.current_epoch % self.print_every == 0:
                avg_loss = total_loss / iterations_per_epoch
                self.loss_history.append(float('%.6f' % avg_loss))
                print('epoch: %d time: %.2f loss: %.6f' % (self.current_epoch, end - start, avg_loss))

                self.model.eval()
                train_acc = self.check_accuracy(self.X_train, self.y_train, batch_size=self.batch_size)
                val_acc = self.check_accuracy(self.X_val, self.y_val, batch_size=self.batch_size)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                print('train acc: %.4f; val_acc: %.4f' % (train_acc, val_acc))

                if val_acc >= self.best_val_acc and train_acc > self.best_train_acc:
                    self.best_train_acc = train_acc
                    self.best_val_acc = val_acc
                    self.best_params = dict()
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
