# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:32
# @Author  : zj


from .pool2row import *
from .Layer import *

__all__ = ['FC']


class FC(Layer):
    """
    fully connected layer
    全连接层
    """

    def __init__(self, num_in, num_out, momentum=0, nesterov=False):
        """
        :param num_in: 前一层神经元个数
        :param num_out: 当前层神经元个数
        :param momentum: 动量因子
        :param nesterov: 是否使用Nesterov加速梯度
        """
        super(FC, self).__init__()
        assert isinstance(num_in, int) and num_in > 0
        assert isinstance(num_out, int) and num_out > 0

        self.W = {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(num_in, num_out)),
                  'grad': 0,
                  'v': 0,
                  'momentum': momentum,
                  'nesterov': nesterov}
        self.b = {'val': 0.01 * np.random.normal(loc=0, scale=1.0, size=(1, num_out)), 'grad': 0}
        self.inputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape == [N, num_in]
        assert len(inputs.shape) == 2
        self.inputs = inputs.copy()

        z = inputs.dot(self.W['val']) + self.b['val']
        return z

    def backward(self, grad_out):
        self.W['grad'] = self.inputs.T.dot(grad_out)
        self.b['grad'] = np.sum(grad_out, axis=0, keepdims=True) / grad_out.shape[0]

        grad_in = grad_out.dot(self.W['val'].T)
        return grad_in

    def update(self, learning_rate=0, regularization_rate=0):
        v_prev = self.W['v']
        self.W['v'] = self.W['momentum'] * self.W['v'] - learning_rate * (
                self.W['grad'] + regularization_rate * self.W['val'])
        if self.W['nesterov']:
            self.W['val'] += (1 + self.W['momentum']) * self.W['v'] - self.W['momentum'] * v_prev
        else:
            self.W['val'] += self.W['v']
        self.b['val'] -= learning_rate * self.b['grad']

    def get_params(self):
        return {'W': self.W['val'], 'momentum': self.W['momentum'], 'nesterov': self.W['nesterov'], 'b': self.b['val']}

    def set_params(self, params):
        self.W['val'] = params.get('W')
        self.b['val'] = params.get('b')

        self.W['momentum'] = params.get('momentum', 0.0)
        self.W['nesterov'] = params.get('nesterov', False)
