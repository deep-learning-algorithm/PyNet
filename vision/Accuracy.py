# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:30
# @Author  : zj


import numpy as np
import torch


class Accuracy(object):

    def __init__(self):
        pass

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        predicted = np.argmax(scores, axis=1)
        return np.mean(predicted == labels), predicted

    def compute_v2(self, data_array, labels_array, net, batch_size=128):
        total_accuracy = 0.0
        num = 0

        range_list = np.arange(0, data_array.shape[0] - batch_size, step=batch_size)
        for i in range_list:
            data = data_array[i:i + batch_size]
            labels = labels_array[i:i + batch_size]

            scores = net.predict(data)
            predicted = np.argmax(scores, axis=1)
            total_accuracy += np.mean(predicted == labels)
            num += 1

        return total_accuracy / num

    def compute_pytorch(self, loader, net, device):
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
