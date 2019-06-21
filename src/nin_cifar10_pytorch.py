# -*- coding: utf-8 -*-

# @Time    : 19-6-21 下午3:41
# @Author  : zj

import torch
import torch.nn as nn
import torch.optim as optim
import time
import vision.data
import models.pytorch

epochs = 100
batch_size = 128
lr = 1e-3
momentum = 0.9

data_path = '/home/lab305/Documents/data/cifar_10'


def train():
    train_loader, test_loader = vision.data.load_cifar10_pytorch(data_path, batch_size=batch_size, shuffle=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    net = models.pytorch.nin(in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # stepLR = StepLR(optimer, 100, 0.5)

    best_train_accuracy = 0.995
    best_test_accuracy = 0

    accuracy = vision.Accuracy()

    loss_list = []
    train_list = []
    for i in range(epochs):
        num = 0
        total_loss = 0
        start = time.time()
        # 训练阶段
        net.train()
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
        # stepLR.step()

        avg_loss = total_loss / num
        loss_list.append(float('%.8f' % avg_loss))
        print('epoch: %d time: %.2f loss: %.8f' % (i + 1, end - start, avg_loss))

        if i % 20 == 19:
            # 验证阶段
            net.eval()
            train_accuracy = accuracy.compute_pytorch(train_loader, net, device)
            train_list.append(float('%.4f' % train_accuracy))
            if best_train_accuracy < train_accuracy:
                best_train_accuracy = train_accuracy

                test_accuracy = accuracy.compute_pytorch(test_loader, net, device)
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy

            print('best train accuracy: %.2f %%   best test accuracy: %.2f %%' % (
                best_train_accuracy * 100, best_test_accuracy * 100))
            print(loss_list)
            print(train_list)


if __name__ == '__main__':
    start = time.time()
    train()
    end = time.time()
    print('training need time: %.3f' % (end - start))
