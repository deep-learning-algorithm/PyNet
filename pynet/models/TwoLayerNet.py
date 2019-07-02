# -*- coding: utf-8 -*-

# @Time    : 19-6-29 下午8:28
# @Author  : zj


from pynet import nn
from .Net import Net
from .utils import load_params

__all__ = ['TwoLayerNet']

model_urls = {
    'two_layer_net': ''
}


class TwoLayerNet(Net):
    """
    实现2层神经网络
    """

    def __init__(self, num_in=0, num_hidden=0, num_out=0, dropout=1.0, weight_scale=1e-2):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.FC(num_in, num_hidden, weight_scale)
        self.fc2 = nn.FC(num_hidden, num_out, weight_scale)

        self.relu = nn.ReLU()
        self.z1 = None
        self.z1_cache = None
        self.z2_cache = None

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
        # inputs.shape = [N, D_in]
        assert len(inputs.shape) == 2
        self.z1, self.z1_cache = self.fc1(inputs, self.params['W1'], self.params['b1'])
        a1 = self.relu(self.z1)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            self.U1 = self.dropout(a1.shape, self.dropout_param['p'])
            a1 *= self.U1

        z2, self.z2_cache = self.fc2(a1, self.params['W2'], self.params['b2'])

        return z2

    def backward(self, grad_out):
        grad = dict()
        grad['W2'], grad['b2'], da1 = self.fc2.backward(grad_out, self.z2_cache)
        if self.use_dropout and self.dropout_param['mode'] == 'train':
            da1 *= self.U1

        dz1 = self.relu.backward(da1, self.z1)
        grad['W1'], grad['b1'], da0 = self.fc1.backward(dz1, self.z1_cache)

        return grad

    def _get_params(self):
        params = dict()
        params['W1'], params['b1'] = self.fc1.get_params()
        params['W2'], params['b2'] = self.fc2.get_params()
        return params

    def train(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'train'

    def eval(self):
        if self.use_dropout:
            self.dropout_param['mode'] = 'test'


def two_layer_net(pretrained=False, **kwargs):
    """
    创建模型对象
    """

    model = TwoLayerNet(**kwargs)
    if pretrained:
        params = load_params(model_urls['alexnet'])
        model.set_params(params)
    return model
