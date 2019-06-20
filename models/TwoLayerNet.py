# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:08
# @Author  : zj

import nn
from .Net import Net
from .utils import load_params

__all__ = ['TwoLayerNet', 'two_layer_net']

model_urls = {
    'two_layer_net': ''
}


class TwoLayerNet(Net):
    """
    实现2层神经网络
    """

    def __init__(self, num_in, num_hidden, num_out):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.FC(num_in, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.FC(num_hidden, num_out)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, D_in]
        assert len(inputs.shape) == 2
        a1 = self.relu(self.fc1(inputs))
        z2 = self.fc2(a1)

        return z2

    def backward(self, grad_out):
        da1 = self.fc2.backward(grad_out)
        dz1 = self.relu.backward(da1)
        da0 = self.fc1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.fc2.update(lr, reg)
        self.fc1.update(lr, reg)

    def get_params(self):
        return {'fc1': self.fc1.get_params(), 'fc2': self.fc2.get_params()}

    def set_params(self, params):
        self.fc1.set_params(params['fc1'])
        self.fc2.set_params(params['fc2'])


def two_layer_net(num_in, num_hidden, num_out, pretrained=False):
    """
    创建模型对象
    """

    model = TwoLayerNet(num_in, num_hidden, num_out)
    if pretrained:
        params = load_params(model_urls['two_layer_net'])
        model.set_params(params)
    return model
