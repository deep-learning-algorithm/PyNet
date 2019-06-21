# -*- coding: utf-8 -*-

# @Time    : 19-6-21 下午2:56
# @Author  : zj

import torch
import torch.nn as nn

__all__ = ['NIN', 'nin']

model_urls = {
    'nin': ''
}


class NIN(nn.Module):

    def __init__(self, in_channels=1, out_channels=10):
        super(NIN, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 160, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(160, 96, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d()
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 192, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, 192, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d()
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 192, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, out_channels, (1, 1), stride=1, padding=0),
            nn.ReLU(),
        )

        self.gap = nn.AvgPool2d(8)

    def forward(self, inputs):
        x = self.features1(inputs)
        x = self.features2(x)
        x = self.features3(x)
        x = self.gap(x)

        return x.view(x.shape[0], x.shape[1])


def nin(pretrained=False, **kwargs):
    """
    创建模型对象
    """

    model = NIN(**kwargs)
    # if pretrained:
        # params = load_params(model_urls['nin'])
        # model.set_params(params)
    return model
