# -*- coding: utf-8 -*-

# @Time    : 19-7-2 上午10:14
# @Author  : zj


from pynet import nn
from .Net import Net
from .utils import load_params

__all__ = ['LeNet5']

model_urls = {
    'lenets': ''
}


class LeNet5(Net):
    """
    LeNet-5网络
    """

    def __init__(self, in_channels=1, out_channels=10, dropout=1.0, weight_scale=1e-2):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d2(in_channels, 5, 5, 6, stride=1, padding=0, weight_scale=weight_scale)
        self.conv2 = nn.Conv2d2(6, 5, 5, 16, stride=1, padding=0, weight_scale=weight_scale)
        self.conv3 = nn.Conv2d2(16, 5, 5, 120, stride=1, padding=0)

        self.maxPool1 = nn.MaxPool2(2, 2, 6, stride=2)
        self.maxPool2 = nn.MaxPool2(2, 2, 16, stride=2)
        self.fc1 = nn.FC(120, 84)
        self.fc2 = nn.FC(84, out_channels)

        self.relu = nn.ReLU()
        self.z1 = None
        self.z1_cache = None
        self.z2 = None
        self.z2_cache = None
        self.z3 = None
        self.z3_cache = None
        self.z4 = None
        self.z4_cache = None
        self.z5 = None
        self.z5_cache = None
        self.z6 = None
        self.z6_cache = None
        # self.z7 = None
        self.z7_cache = None

        self.use_dropout = dropout != 1.0
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'
            self.dropout_param['p'] = dropout
            self.dropout = nn.Dropout()
            self.U1 = None

        self.params = self._get_params()

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        self.z1, self.z1_cache = self.conv1(inputs, self.params['W1'], self.params['b1'])
        a1 = self.relu(self.z1)
        self.z2, self.z2_cache = self.maxPool1(a1)

        self.z3, self.z3_cache = self.conv2(self.z2, self.params['W2'], self.params['b2'])
        a3 = self.relu(self.z3)
        self.z4, self.z4_cache = self.maxPool2(a3)

        self.z5, self.z5_cache = self.conv3(self.z4, self.params['W3'], self.params['b3'])
        a5 = self.relu(self.z5)

        # (N, C, 1, 1) -> (N, C)
        a5 = a5.reshape(a5.shape[0], -1)

        self.z6, self.z6_cache = self.fc1(a5, self.params['W4'], self.params['b4'])
        a6 = self.relu(self.z6)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            self.U1 = self.dropout(a6.shape, self.dropout_param['p'])
            a6 *= self.U1

        z7, self.z7_cache = self.fc2(a6, self.params['W5'], self.params['b5'])

        return z7

    def backward(self, grad_out):
        grad = dict()

        grad['W5'], grad['b5'], da6 = self.fc2.backward(grad_out, self.z7_cache)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            da6 *= self.U1

        dz6 = self.relu.backward(da6, self.z6)
        grad['W4'], grad['b4'], da5 = self.fc1.backward(dz6, self.z6_cache)

        # [N, C] -> [N, C, 1, 1]
        N, C = da5.shape[:2]
        da5 = da5.reshape(N, C, 1, 1)

        dz5 = self.relu.backward(da5, self.z5)
        grad['W3'], grad['b3'], dz4 = self.conv3.backward(dz5, self.z5_cache)

        da3 = self.maxPool2.backward(dz4, self.z4_cache)
        dz3 = self.relu.backward(da3, self.z3)
        grad['W2'], grad['b2'], da2 = self.conv2.backward(dz3, self.z3_cache)

        da1 = self.maxPool1.backward(da2, self.z2_cache)
        dz1 = self.relu.backward(da1, self.z1)
        grad['W1'], grad['b1'], da0 = self.conv1.backward(dz1, self.z1_cache)

        return grad

    def _get_params(self):
        params = dict()
        params['W1'], params['b1'] = self.conv1.get_params()
        params['W2'], params['b2'] = self.conv2.get_params()
        params['W3'], params['b3'] = self.conv3.get_params()
        params['W4'], params['b4'] = self.fc1.get_params()
        params['W5'], params['b5'] = self.fc2.get_params()

        return params

    def train(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'

    def eval(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'test'
