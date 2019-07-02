# -*- coding: utf-8 -*-

# @Time    : 19-7-1 下午8:04
# @Author  : zj

import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if isinstance(lr, float) and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if nesterov is True and momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        # self.params = params
        self.use_momentum = momentum != 0
        self.nesterov = nesterov
        defaults = dict(lr=lr, momentum=momentum)

        super(SGD, self).__init__(params, defaults)

        # self.optim_configs = {}
        # for p in params.keys():
        #     d = {k: v for k, v in defaults.items()}
        #     self.optim_configs[p] = d

    def step(self, grad):
        assert isinstance(grad, dict)
        for p, w in self.params.items():
            dw = grad[p]
            config = self.optim_configs[p]
            next_w, next_config = self._sgd(w, dw, config, is_weight=('W' in p))
            self.params[p] = next_w
            self.optim_configs[p] = next_config

    def _sgd(self, w, dw, config, is_weight=True):

        if self.use_momentum and is_weight:
            v_prev = config.get('velocity', np.zeros_like(w))

            if self.nesterov:
                v = config['momentum'] * v_prev - config['lr'] * dw
                next_w = w + (1 + config['momentum']) * v - config['momentum'] * v_prev
            else:
                v = config['momentum'] * v_prev - config['lr'] * dw
                next_w = w + v

            config['velocity'] = v
        else:
            next_w = w - config['lr'] * dw

        return next_w, config
