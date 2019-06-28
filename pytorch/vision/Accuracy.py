# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:30
# @Author  : zj

import torch


class Accuracy(object):

    def __init__(self):
        pass

    def __call__(self, loader, net, device):
        return self.forward(loader, net, device)

    def forward(self, loader, net, device):
        total_accuracy = 0
        num = 0
        for item in loader:
            data, labels = item
            data = data.to(device)
            labels = labels.to(device)

            scores = net.forward(data)
            predicted = torch.argmax(scores, dim=1)
            total_accuracy += torch.mean((predicted == labels).float()).item()
            num += 1
        return total_accuracy / num
