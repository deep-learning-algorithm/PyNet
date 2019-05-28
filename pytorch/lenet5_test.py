# -*- coding: utf-8 -*-

# @Time    : 19-5-28 上午10:37
# @Author  : zj

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
import time

# 批量大小
batch_size = 128
# 学习率
learning_rate = 1e-3
# # 动量
# momentum = 0.9
# 迭代次数
epoches = 500


def load_mnist_data(batch_size=128, shuffle=False):
    data_dir = '/home/zj/data/'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_data_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.fc1 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc2 = nn.Linear(84, 10, bias=True)

    def forward(self, input):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def compute_accuracy(loader, net, device):
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


def draw(loss_list, title='损失图', ylabel='损失值', xlabel='迭代/次'):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(loss_list)
    plt.show()


def classify_mnist():
    train_loader, test_loader = load_mnist_data(batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # optimer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    optimer = optim.SGD(net.parameters(), lr=learning_rate)

    loss_list = []
    for i in range(epoches):
        num = 0
        total_loss = 0
        start = time.time()
        for j, item in enumerate(train_loader, 0):
            data, labels = item
            data = data.to(device)
            labels = labels.to(device)

            scores = net.forward(data)
            loss = criterion.forward(scores, labels)
            total_loss += loss.item()

            optimer.zero_grad()
            loss.backward()
            optimer.step()
            num += 1
        end = time.time()
        print('one epoch need time: %f' % (end - start))

        total_loss /= num
        print('epoch: %d loss: %.6f' % (i + 1, total_loss / num))
        loss_list.append(total_loss / num)
        if (i % 50) == 49:
            draw(loss_list)
            train_accuracy = compute_accuracy(test_loader, net, device)
            test_accuracy = compute_accuracy(test_loader, net, device)
            print('train accuracy: %f test accuracy: %f' % (train_accuracy, test_accuracy))


if __name__ == '__main__':
    classify_mnist()
