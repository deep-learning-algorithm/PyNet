# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:20
# @Author  : zj


import nn
from .Net import Net
from .utils import load_params


__all__ = ['LeNet5', 'lenet5']


model_urls = {
    'lenets': ''
}


class LeNet5(Net):
    """
    LeNet-5网络
    """

    def __init__(self, in_channels, momentum=0, nesterov=False, p_h=1.0):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, 5, 6, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.conv2 = nn.Conv2d(6, 5, 5, 16, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.conv3 = nn.Conv2d(16, 5, 5, 120, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.maxPool1 = nn.MaxPool(2, 2, 6, stride=2)
        self.maxPool2 = nn.MaxPool(2, 2, 16, stride=2)
        self.fc1 = nn.FC(120, 84, momentum=momentum, nesterov=nesterov)
        self.fc2 = nn.FC(84, 10, momentum=momentum, nesterov=nesterov)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout()

        self.p_h = p_h
        self.U1 = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.maxPool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxPool2(x)
        x = self.relu3(self.conv3(x))

        # (N, C, 1, 1) -> (N, C)
        x = x.reshape(x.shape[0], -1)
        x = self.relu4(self.fc1(x))
        self.U1 = self.dropout(x.shape, self.p_h)
        x *= self.U1

        x = self.fc2(x)

        return x

    def backward(self, grad_out):
        da6 = self.fc2.backward(grad_out)
        da6 *= self.U1

        dz6 = self.relu4.backward(da6)
        da5 = self.fc1.backward(dz6)
        # [N, C] -> [N, C, 1, 1]
        N, C = da5.shape[:2]
        da5 = da5.reshape(N, C, 1, 1)
        dz5 = self.relu3.backward(da5)
        da4 = self.conv3.backward(dz5)

        dz4 = self.maxPool2.backward(da4)
        dz3 = self.relu2.backward(dz4)
        da2 = self.conv2.backward(dz3)

        da1 = self.maxPool1.backward(da2)
        dz1 = self.relu1.backward(da1)
        self.conv1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.fc2.update(learning_rate=lr, regularization_rate=reg)
        self.fc1.update(learning_rate=lr, regularization_rate=reg)
        self.conv3.update(learning_rate=lr, regularization_rate=reg)
        self.conv2.update(learning_rate=lr, regularization_rate=reg)
        self.conv1.update(learning_rate=lr, regularization_rate=reg)

    def predict(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.maxPool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxPool2(x)
        x = self.relu3(self.conv3(x))
        # (N, C, 1, 1) -> (N, C)
        x = x.reshape(x.shape[0], -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_params(self):
        out = dict()
        out['conv1'] = self.conv1.get_params()
        out['conv2'] = self.conv2.get_params()
        out['conv3'] = self.conv3.get_params()

        out['fc1'] = self.fc1.get_params()
        out['fc2'] = self.fc2.get_params()

        out['p_h'] = self.p_h

        return out

    def set_params(self, params):
        self.conv1.set_params(params['conv1'])
        self.conv2.set_params(params['conv2'])
        self.conv3.set_params(params['conv3'])

        self.fc1.set_params(params['fc1'])
        self.fc2.set_params(params['fc2'])

        self.p_h = params.get('p_h', 1.0)


def lenet5(pretrained=False, **kwargs):
    """
    创建模型对象
    """

    model = LeNet5(**kwargs)
    if pretrained:
        params = load_params(model_urls['two_layer_net'])
        model.set_params(params)
    return model
