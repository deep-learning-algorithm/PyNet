# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午7:22
# @Author  : zj

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def load_cifar10(cifar10_path, batch_size=128, shuffle=False, dst_size=(227, 227)):
    transform = transforms.Compose([
        transforms.Resize(dst_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data_set = datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)
    test_data_set = datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return train_loader, test_loader
