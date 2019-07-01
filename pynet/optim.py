# -*- coding: utf-8 -*-

# @Time    : 19-6-29 下午8:46
# @Author  : zj

import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v

    config['velocity'] = v
    return next_w, config


def sgd_nesterov(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('momentum', 0.9)
    v_prev = config.get('velocity', np.zeros_like(w))

    v = config['momentum'] * v_prev - config['learning_rate'] * dw
    next_w = w + (1 + config['momentum']) * v - config['momentum'] * v_prev

    config['velocity'] = v
    return next_w, config
