# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:18
# @Author  : zj

import nn
from .Net import Net
from .utils import load_params

__all__ = ['ThreeLayerNet', 'three_layer_net']

model_urls = {
    'three_layer_net': ''
}


class ThreeLayerNet(Net):
    """
    实现3层神经网络
    """

    def __init__(self, num_in=0, num_h1=0, num_h2=0, num_out=0, momentum=0, nesterov=False, p_h=1.0):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.FC(num_in, num_h1, momentum=momentum, nesterov=nesterov)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.FC(num_h1, num_h2, momentum=momentum, nesterov=nesterov)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.FC(num_h2, num_out, momentum=momentum, nesterov=nesterov)

        self.dropout = nn.Dropout()

        self.p_h = p_h
        self.U1 = None
        self.U2 = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, D_in]
        assert len(inputs.shape) == 2
        a1 = self.relu1(self.fc1(inputs))
        self.U1 = self.dropout(a1.shape, self.p_h)
        a1 *= self.U1

        a2 = self.relu2(self.fc2(a1))
        self.U2 = self.dropout(a2.shape, self.p_h)
        a2 *= self.U2

        z3 = self.fc3(a2)

        return z3

    def backward(self, grad_out):
        da2 = self.fc3.backward(grad_out) * self.U2
        dz2 = self.relu2.backward(da2)
        da1 = self.fc2.backward(dz2) * self.U1
        dz1 = self.relu1.backward(da1)
        da0 = self.fc1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.fc3.update(learning_rate=lr, regularization_rate=reg)
        self.fc2.update(learning_rate=lr, regularization_rate=reg)
        self.fc1.update(learning_rate=lr, regularization_rate=reg)

    def predict(self, inputs):
        # inputs.shape = [N, D_in]
        assert len(inputs.shape) == 2
        a1 = self.relu1(self.fc1(inputs))
        a2 = self.relu2(self.fc2(a1))
        z3 = self.fc3(a2)

        return z3

    def get_params(self):
        return {'fc1': self.fc1.get_params(), 'fc2': self.fc2.get_params(), 'fc3': self.fc3.get_params(),
                'p_h': self.p_h}

    def set_params(self, params):
        self.fc1.set_params(params['fc1'])
        self.fc2.set_params(params['fc2'])
        self.fc3.set_params(params['fc3'])
        self.p_h = params.get('p_h', 1.0)


def three_layer_net(pretrained=False, **kwargs):
    """
    创建模型对象
    """

    model = ThreeLayerNet(**kwargs)
    if pretrained:
        params = load_params(model_urls['three_layer_net'])
        model.set_params(params)
    return model
