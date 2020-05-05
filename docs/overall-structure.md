
# 整体架构

## 整个仓库

```
├── CHANGELOG.md
├── docs
├── examples
├── imgs
│   ├── logo2.png
│   └── logo.png
├── LICENSE
├── mkdocs.yml
├── pynet
├── README.EN.md
├── README.md
└── requirements.txt
```

* `PyNet`实现代码位于`pynet`
* `PyNet`示例代码位于`examples`
* 说明文档位于`docs`

## pynet

```
├── __init__.py
├── models
├── nn
├── optim
├── __pycache__
├── requirements.txt
├── solver.py
└── vision
```

* 层的实现位于`nn`，比如`Conv2d/MaxPool/Dropout/BN/ReLU/...`
* 模型的实现位于`models`，比如`FCNet/LeNet5/AlexNet/NIN`
* 优化器和学习率调度器的实现位于`optim`，比如`SGD/StepLR`
* 数据加载和训练结果绘制的实现位于`vision`
* 求解器实现位于`solver.py`

## examples

```
├── 2_nn_mnist.py
├── 3_nn_cifar10.py
├── 3_nn_iris.py
├── 3_nn_mnist.py
├── 3_nn_orl.py
├── lenet5_mnist.py
└── nin_cifar10.py
```

测试了`4`个网络模型：`2`层`/3`层神经网络、`LeNet5`和`NIN`；以及`4`个数据集：`MNIST、CIFAR10、iris`和`ORL`

*`Note`：使用`NIN`的使用就很慢了*