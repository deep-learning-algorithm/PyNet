# -*- coding: utf-8 -*-

# @Time    : 19-6-21 上午11:00
# @Author  : zj


import pynet.nn as nn
from .Net import Net
from .utils import load_params

__all__ = ['NIN', 'nin']

model_urls = {
    'nin': ''
}


class NIN(Net):
    """
    NIN网络
    """

    def __init__(self, in_channels=1, out_channels=10, momentum=0, nesterov=False, p_h=1.0):
        super(NIN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, 5, 192, stride=1, padding=2, momentum=momentum, nesterov=nesterov)
        self.conv2 = nn.Conv2d(96, 5, 5, 192, stride=1, padding=2, momentum=momentum, nesterov=nesterov)
        self.conv3 = nn.Conv2d(192, 3, 3, 192, stride=1, padding=1, momentum=momentum, nesterov=nesterov)

        self.mlp1 = nn.Conv2d(192, 1, 1, 160, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.mlp2 = nn.Conv2d(160, 1, 1, 96, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.mlp2_1 = nn.Conv2d(192, 1, 1, 192, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.mlp2_2 = nn.Conv2d(192, 1, 1, 192, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.mlp3_1 = nn.Conv2d(192, 1, 1, 192, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.mlp3_2 = nn.Conv2d(192, 1, 1, out_channels, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.maxPool1 = nn.MaxPool(2, 2, 96, stride=2)
        self.maxPool2 = nn.MaxPool(2, 2, 192, stride=2)

        self.gap = nn.GAP()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()

        self.dropout = nn.Dropout2d()

        self.p_h = p_h
        self.U1 = None
        self.U2 = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.relu2(self.mlp1(x))
        x = self.relu3(self.mlp2(x))
        x = self.maxPool1(x)
        self.U1 = self.dropout(x.shape, self.p_h)
        x *= self.U1

        x = self.relu4(self.conv2(x))
        x = self.relu5(self.mlp2_1(x))
        x = self.relu6(self.mlp2_2(x))
        x = self.maxPool2(x)
        self.U2 = self.dropout(x.shape, self.p_h)
        x *= self.U2

        x = self.relu7(self.conv3(x))
        x = self.relu8(self.mlp3_1(x))
        x = self.relu9(self.mlp3_2(x))

        x = self.gap(x)
        return x

    def backward(self, grad_out):
        # grad_out.shape = [N, C]
        assert len(grad_out.shape) == 2
        da11 = self.gap.backward(grad_out)

        dz11 = self.relu9.backward(da11)
        da10 = self.mlp3_2.backward(dz11)
        dz10 = self.relu8.backward(da10)
        da9 = self.mlp3_1.backward(dz10)
        dz9 = self.relu7.backward(da9)
        da8 = self.conv3.backward(dz9)

        da8 *= self.U2
        da7 = self.maxPool2.backward(da8)
        dz7 = self.relu6.backward(da7)
        da6 = self.mlp2_2.backward(dz7)
        dz6 = self.relu5.backward(da6)
        da5 = self.mlp2_1.backward(dz6)
        dz5 = self.relu4.backward(da5)
        da4 = self.conv2.backward(dz5)

        da4 *= self.U1
        da3 = self.maxPool1.backward(da4)
        dz3 = self.relu3.backward(da3)
        da2 = self.mlp2.backward(dz3)
        dz2 = self.relu2.backward(da2)
        da1 = self.mlp1.backward(dz2)
        dz1 = self.relu1.backward(da1)
        da0 = self.conv1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.mlp3_2.update(learning_rate=lr, regularization_rate=reg)
        self.mlp3_1.update(learning_rate=lr, regularization_rate=reg)
        self.conv3.update(learning_rate=lr, regularization_rate=reg)

        self.mlp2_2.update(learning_rate=lr, regularization_rate=reg)
        self.mlp2_1.update(learning_rate=lr, regularization_rate=reg)
        self.conv2.update(learning_rate=lr, regularization_rate=reg)

        self.mlp2.update(learning_rate=lr, regularization_rate=reg)
        self.mlp1.update(learning_rate=lr, regularization_rate=reg)
        self.conv1.update(learning_rate=lr, regularization_rate=reg)

    def predict(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.relu2(self.mlp1(x))
        x = self.relu3(self.mlp2(x))
        x = self.maxPool1(x)

        x = self.relu4(self.conv2(x))
        x = self.relu5(self.mlp2_1(x))
        x = self.relu6(self.mlp2_2(x))
        x = self.maxPool2(x)

        x = self.relu7(self.conv3(x))
        x = self.relu8(self.mlp3_1(x))
        x = self.relu9(self.mlp3_2(x))

        x = self.gap(x)
        return x

    def get_params(self):
        out = dict()
        out['conv1'] = self.conv1.get_params()
        out['conv2'] = self.conv2.get_params()
        out['conv3'] = self.conv3.get_params()

        out['mlp1'] = self.mlp1.get_params()
        out['mlp2'] = self.mlp2.get_params()
        out['mlp2_1'] = self.mlp2_1.get_params()
        out['mlp2_2'] = self.mlp2_2.get_params()
        out['mlp3_1'] = self.mlp3_1.get_params()
        out['mlp3_2'] = self.mlp3_2.get_params()

        out['p_h'] = self.p_h

        return out

    def set_params(self, params):
        self.conv1.set_params(params['conv1'])
        self.conv2.set_params(params['conv2'])
        self.conv3.set_params(params['conv3'])

        self.mlp1.set_params(params['mlp1'])
        self.mlp2.set_params(params['mlp2'])
        self.mlp2_1.set_params(params['mlp2_1'])
        self.mlp2_2.set_params(params['mlp2_1'])
        self.mlp3_1.set_params(params['mlp3_1'])
        self.mlp3_2.set_params(params['mlp3_1'])

        self.p_h = params.get('p_h', 1.0)


def nin(pretrained=False, **kwargs):
    """
    创建模型对象
    """

    model = NIN(**kwargs)
    if pretrained:
        params = load_params(model_urls['nin'])
        model.set_params(params)
    return model
