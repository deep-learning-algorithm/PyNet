
![](imgs/logo.png)

#  Numpy-based deep learning library 

[中文版本（Chinese version）](./README.zh-CN.md)

Implementation of deep learning based on numpy, modular design guarantees easy implementation of the model, which is suitable for the introduction of junior researchers in deep learning.

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
* Fully-Connected Layer (FC)
* Max-Pooling layer (MaxPool)
* ReLU Layer (ReLU)
* Random Dropout Layer (Dropout/Dropout2d)
* Softmax
* Cross Entropy Loss
* Gloabl Average Pool (GAP)

## Catalog

```
.
├── examples                          # pynet use examples
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
├── plt                               # draw loss and acc
│   ├── draw_acc.py
│   ├── draw_loss.py
│   ├── __init__.py
│   └── __pycache__
├── pynet                             
│   ├── __init__.py
│   ├── models                        # model definition
│   ├── nn                            # layer definition
│   ├── optim                         # optimizer
│   ├── __pycache__
│   ├── solver.py                     # solver
│   └── vision                        # data correlation
├── pytorch
│   ├── examples
│   ├── __init__.py
│   ├── models
│   └── vision
├── README.md
└── README.zh-CN.md
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags](https://github.com/zjZSTU/PyNet/releases) on this repository.

## Authors

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## License

This project is licensed under the Apache License v2.0 - see the LICENSE file for details

## Acknowledgments

* [cs231n](http://cs231n.github.io/)
* [PyTorch](https://pytorch.org/)