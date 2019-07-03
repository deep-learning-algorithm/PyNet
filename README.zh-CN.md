
![](imgs/logo.png)

# 基于numpy的深度学习库

基于`numpy`的深度学习实现，模块化设计保证模型的轻松实现，适用于深度学习初级研究人员的入门

*同时附带了PyTorch示例*

## 功能特性

已实现网络模型（位于`pynet/models`）：

* 2层神经网络
* 3层神经网络
* LeNet-5
* AlexNet
* NIN

已实现网络层（位于`pynet/nn`）：

* 卷积层
* 全连接层
* 最大池化层
* ReLU
* 随机失活
* Softmax
* 交叉熵损失
* 全局平均池化层

## 工程结构

```
.
├── examples                          # pynet使用示例
│   ├── 2_nn_mnist.py
│   ├── 3_nn_cifar10.py
│   ├── 3_nn_iris.py
│   ├── 3_nn_mnist.py
│   ├── 3_nn_orl.py
│   ├── lenet5_mnist.py
│   └── nin_cifar10.py
├── imgs                              
│   ├── logo2.png
│   └── logo.png
├── LICENSE
├── plt                               # 绘制损失和精度图
│   ├── draw_acc.py
│   ├── draw_loss.py
│   ├── __init__.py
│   └── __pycache__
├── pynet                             
│   ├── __init__.py
│   ├── models                        # 模型定义
│   ├── nn                            # 层定义
│   ├── optim                         # 优化器
│   ├── __pycache__
│   ├── solver.py                     # 求解器
│   └── vision                        # 数据相关
├── pytorch
│   ├── examples
│   ├── __init__.py
│   ├── models
│   └── vision
├── README.md
└── README.zh-CN.md
```

## 版本化

我们使用[SemVer](http://semver.org/)进行版本控制。 对于可用的版本，请参阅仓库中的[标记](https://github.com/zjZSTU/PyNet/releases)

## 作者

* zhujian - *初始工作* - [zjZSTU](https://github.com/zjZSTU)

## 协议

本工程基于`Apache v2.0`协议 - 具体协议内容参考`LICENSE`文件

## 致谢

* [cs231n](http://cs231n.github.io/)
* [PyTorch](https://pytorch.org/)