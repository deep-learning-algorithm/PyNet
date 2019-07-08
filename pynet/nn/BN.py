# -*- coding: utf-8 -*-

# @Time    : 19-7-4 下午2:45
# @Author  : zj

import numpy as np

__all__ = ['BN']


class BN:
    """
    batch normalization layer
    批量归一化层
    """

    def __call__(self, inputs, gamma, beta, bn_param):
        return self.forward(inputs, gamma, beta, bn_param)

    def forward(self, inputs, gamma, beta, bn_param):
        # inputs.shape == [N, C]
        assert len(inputs.shape) == 2

        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = inputs.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=inputs.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=inputs.dtype))

        out, cache = None, None
        if mode == 'train':
            sample_mean = np.mean(inputs, axis=0, keepdims=True)
            sample_var = np.var(inputs, axis=0, keepdims=True)
            x_norm = (inputs - sample_mean) / np.sqrt(sample_var + eps)

            out = x_norm * gamma + beta
            cache = (sample_mean, sample_var, x_norm, gamma, eps, inputs)

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
        elif mode == 'test':
            stds = np.sqrt(running_var + eps)
            out = gamma / stds * inputs + (beta - gamma * running_mean / stds)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        return out, cache

    def backward(self, grad_out, cache):
        sample_mean, sample_var, x_norm, gamma, eps, x = cache
        m = x.shape[0]

        dgamma = np.sum(grad_out * x_norm, axis=0)
        dbeta = np.sum(grad_out, axis=0)
        dnorm = grad_out * gamma

        dvar = dnorm * (x - sample_mean) * (-0.5) * (sample_var + eps) ** (-1.5)
        dvar = np.sum(dvar, axis=0, keepdims=True)
        dmean = np.sum(dnorm * (-1) / np.sqrt(sample_var + eps), axis=0, keepdims=True)
        dmean += dvar * np.sum(-2 * (x - sample_mean), axis=0, keepdims=True) / m

        dx = dnorm / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / m + dmean / m

        return dx, dgamma, dbeta
