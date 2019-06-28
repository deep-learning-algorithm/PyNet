# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:22
# @Author  : zj

from pynet import nn
from .Net import Net
from .utils import load_params

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': ''
}


class AlexNet(Net):
    """
    AlexNet模型
    """

    def __init__(self, in_channels, out_channels, momentum=0, nesterov=False):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 11, 11, 96, stride=4, padding=0, momentum=momentum, nesterov=nesterov)
        self.conv2 = nn.Conv2d(96, 5, 5, 256, stride=1, padding=2, momentum=momentum, nesterov=nesterov)
        self.conv3 = nn.Conv2d(256, 3, 3, 384, stride=1, padding=1, momentum=momentum, nesterov=nesterov)
        self.conv4 = nn.Conv2d(384, 3, 3, 384, stride=1, padding=1, momentum=momentum, nesterov=nesterov)
        self.conv5 = nn.Conv2d(384, 3, 3, 256, stride=1, padding=1, momentum=momentum, nesterov=nesterov)

        self.maxPool1 = nn.MaxPool(3, 3, 96, stride=2)
        self.maxPool2 = nn.MaxPool(3, 3, 256, stride=2)
        self.maxPool3 = nn.MaxPool(3, 3, 256, stride=2)
        self.fc1 = nn.FC(9216, 4096, momentum=momentum, nesterov=nesterov)
        self.fc2 = nn.FC(4096, 4096, momentum=momentum, nesterov=nesterov)
        self.fc3 = nn.FC(4096, out_channels, momentum=momentum, nesterov=nesterov)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.dropout = nn.Dropout()

        self.U1 = None
        self.U2 = None

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
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.maxPool3(x)

        # (N, C, 1, 1) -> (N, C)
        x = x.reshape(x.shape[0], -1)
        x = self.relu6(self.fc1(x))
        self.U1 = self.dropout(x.shape, 0.5)
        x *= self.U1
        x = self.relu7(self.fc2(x))
        self.U2 = self.dropout(x.shape, 0.5)
        x *= self.U2

        x = self.fc3(x)
        return x

    def backward(self, grad_out):
        da10 = self.fc3.backward(grad_out)
        da10 *= self.U2

        dz10 = self.relu7.backward(da10)
        da9 = self.fc2.backward(dz10)
        da9 *= self.U1

        dz9 = self.relu6.backward(da9)
        da8 = self.fc1.backward(dz9)

        # [N, C] -> [N, C, 1, 1]
        N, C = da8.shape[:2]
        dz8 = da8.reshape(N, C, 1, 1)

        da7 = self.maxPool3.backward(dz8)
        dz7 = self.relu5.backward(da7)
        da6 = self.conv5.backward(dz7)

        dz6 = self.relu4.backward(da6)
        da5 = self.conv4.backward(dz6)

        dz5 = self.relu3.backward(da5)
        dz4 = self.conv3.backward(dz5)

        da3 = self.maxPool2.backward(dz4)
        dz3 = self.relu2.backward(da3)
        dz2 = self.conv2.backward(dz3)

        da1 = self.maxPool1.backward(dz2)
        dz1 = self.relu1.backward(da1)
        self.conv1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.fc3.update(learning_rate=lr, regularization_rate=reg)
        self.fc2.update(learning_rate=lr, regularization_rate=reg)
        self.fc1.update(learning_rate=lr, regularization_rate=reg)
        self.conv5.update(learning_rate=lr, regularization_rate=reg)
        self.conv4.update(learning_rate=lr, regularization_rate=reg)
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
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.maxPool3(x)

        # (N, C, 1, 1) -> (N, C)
        x = x.reshape(x.shape[0], -1)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc1(x))

        x = self.fc2(x)
        return x

    def get_params(self):
        pass

    def set_params(self, params):
        pass


def alexnet(pretrained=False, **kwargs):
    """
    创建模型对象
    """

    model = AlexNet(**kwargs)
    if pretrained:
        params = load_params(model_urls['alexnet'])
        model.set_params(params)
    return model
