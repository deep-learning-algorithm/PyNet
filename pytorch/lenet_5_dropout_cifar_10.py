# -*- coding: utf-8 -*-

# @Time    : 19-6-7 下午3:09
# @Author  : zj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import time

# 批量大小
batch_size = 256
# 迭代次数
epochs = 1000

# 学习率
lr = 1e-2
# 失活率
p_h = 0.5


def load_cifar_10_data(batch_size=128, shuffle=False):
    data_dir = '/home/lab305/Documents/data/cifar_10/'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


class LeNet5(nn.Module):

    def __init__(self, in_channels, p=0.0):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.fc1 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc2 = nn.Linear(84, 10, bias=True)

        self.p = p
        self.dropout2d = nn.Dropout2d(p=p)
        self.dropout = nn.Dropout(p=p)

    def forward(self, inputs):
        a1 = F.relu(self.conv1(inputs))
        a1 = self.dropout2d(a1)
        z2 = self.pool(a1)

        a3 = F.relu(self.conv2(z2))
        a3 = self.dropout2d(a3)
        z4 = self.pool(a3)

        a5 = F.relu(self.conv3(z4))
        a5 = self.dropout2d(a5)

        x = a5.view(-1, self.num_flat_features(a5))

        a6 = F.relu(self.fc1(x))
        a6 = self.dropout(a6)
        return self.fc2(a6)

    def predict(self, inputs):
        a1 = F.relu(self.conv1(inputs))
        z2 = self.pool(a1)

        a3 = F.relu(self.conv2(z2))
        z4 = self.pool(a3)

        a5 = F.relu(self.conv3(z4))

        x = a5.view(-1, self.num_flat_features(a5))

        a6 = F.relu(self.fc1(x))
        return self.fc2(a6)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def compute_accuracy(loader, net, device):
    total = 0
    correct = 0
    for item in loader:
        data, labels = item
        data = data.to(device)
        labels = labels.to(device)

        scores = net.predict(data)
        predicted = torch.argmax(scores, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


def draw(loss_list, title='损失图', ylabel='损失值', xlabel='迭代/100次'):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    train_loader, test_loader = load_cifar_10_data(batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = LeNet5(3, p=p_h).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    stepLR = lr_scheduler.StepLR(optimer, step_size=150, gamma=0.5)

    best_train_accuracy = 0.99
    best_test_accuracy = 0

    loss_list = []
    train_list = []
    for i in range(epochs):
        num = 0
        total_loss = 0
        start = time.time()
        net.train()  # 训练模式
        for j, item in enumerate(train_loader, 0):
            data, labels = item
            data = data.to(device)
            labels = labels.to(device)

            scores = net.forward(data)
            loss = criterion.forward(scores, labels)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            total_loss += loss.item()
            num += 1
        end = time.time()
        stepLR.step()

        avg_loss = total_loss / num
        loss_list.append(float('%.4f' % avg_loss))
        print('epoch: %d time: %.2f loss: %.4f' % (i + 1, end - start, avg_loss))

        if i % 20 == 19:
            # 计算训练数据集检测精度
            net.eval()  # 测试模式
            train_accuracy = compute_accuracy(train_loader, net, device)
            train_list.append(float('%.4f' % train_accuracy))
            if best_train_accuracy < train_accuracy:
                best_train_accuracy = train_accuracy

                test_accuracy = compute_accuracy(test_loader, net, device)
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy

            print('best train accuracy: %.2f %%   best test accuracy: %.2f %%' % (
                best_train_accuracy * 100, best_test_accuracy * 100))
            print(loss_list)
            print(train_list)
