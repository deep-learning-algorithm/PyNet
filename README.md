
![](./logo.png)

# 基于numpy的深度学习库

工程结构：

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
└── README.md

10 directories, 12 files
```

当前实现内容：

1. 神经网络
2. 卷积神经网络
3. 动量加速和`Nesterov`加速
4. 随机失活

实现网络：

1. 2层神经网络
2. 3层神经网络
3. LeNet-5
4. AlexNet
5. NIN

数据集操作：

1. `mnist`
2. `cifar-10`
3. `orl`
4. `iris`
5. `xor`
