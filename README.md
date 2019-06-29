
![](./logo.png)

#  Numpy-based deep learning library 

[中文版本（Chinese version）]()

Convolutional neural network based on numpy, modular design guarantees easy implementation of the model, which is suitable for the introduction of junior researchers in deep learning.

*A PyTorch example is also included.*

## Features

Realized Network Model（Located on the pynet/models）：

* 2-Layer Neural Network
* 3-Layer Neural Network
* LeNet-5
* AlexNet
* NIN

Realized Network Layer（Located on the pynet/nn）：

* Convolution Layer (Conv2d)
* Fully-Connection Layer (FC)
* Max-Pooling layer (MaxPool)
* ReLU Layer (ReLU)
* Random Dropout Layer (Dropout/Dropout2d)
* Softmax
* Cross Entropy Loss
* Gloabl Average Pool (GAP)

## Catalog

```
.
├── examples                          # pynet使用示例
│   ├── 2_nn_xor.py
│   ├── 3_nn_cifar10.py
│   ├── 3_nn_iris.py
│   ├── 3_nn_orl.py
│   ├── lenet5_mnist.py
│   ├── nin_cifar10.py
│   └── nin_cifar10_pytorch.py
├── plt                               # 绘图相关（待调整）
│   ├── anneal_plt.py
│   ├── lenet5_plt.py
│   └── plt.py
├── pynet                             # PyNet库
│   ├── __init__.py
│   ├── models                        # 模型定义
│   ├── nn                            # 层定义
│   └── vision                        # 数据操作
├── pytorch                           # PyTorch使用示例
│   ├── examples                      
│   ├── models                        # 模型定义
│   └── vision                        # 数据操作
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the tags on this repository.

## Authors

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## License

This project is licensed under the Apache License v2.0 - see the LICENSE file for details