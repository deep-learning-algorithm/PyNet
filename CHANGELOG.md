# [](https://github.com/zjZSTU/PyNet/compare/v0.3.0...v) (2019-09-11)



# [0.3.0](https://github.com/zjZSTU/PyNet/compare/v0.2.1...v0.3.0) (2019-09-11)


### Features

* **model:** 实现自定义层数和大小的神经网络FCNet ([c80706d](https://github.com/zjZSTU/PyNet/commit/c80706d))
* **nn:** 实现批量归一化层，在FCNet上实现BN，使用3层网络进行测试 ([c31a014](https://github.com/zjZSTU/PyNet/commit/c31a014))



## [0.2.1](https://github.com/zjZSTU/PyNet/compare/v0.2.0...v0.2.1) (2019-07-03)



# [0.2.0](https://github.com/zjZSTU/PyNet/compare/v0.1.0...v0.2.0) (2019-07-02)


### Bug Fixes

* **optim:** 修复学习率调度器失败 ([c483601](https://github.com/zjZSTU/PyNet/commit/c483601))



# [0.1.0](https://github.com/zjZSTU/PyNet/compare/dba4057...v0.1.0) (2019-06-29)


### Bug Fixes

* **LeNet5:** 批量大小为1时LeNet5前向操作中C5->F6步骤出错 ([48b5a46](https://github.com/zjZSTU/PyNet/commit/48b5a46))
* **MaxPool:** 变量名出错 input->inputs ([af183c2](https://github.com/zjZSTU/PyNet/commit/af183c2))
* **models:** 修复加载错误模型 ([d8c35c8](https://github.com/zjZSTU/PyNet/commit/d8c35c8))


### Features

* **data:** vision包下新增data包，用于数据加载 ([5d82ac0](https://github.com/zjZSTU/PyNet/commit/5d82ac0))
* **data:** 修改.gitignore，添加data目录 ([a243e20](https://github.com/zjZSTU/PyNet/commit/a243e20))
* **dropout:** 实现随机失活操作，修改ThreeNet和LeNet5实现随机失活操作 ([eaef5d9](https://github.com/zjZSTU/PyNet/commit/eaef5d9))
* **GAP:** 实现全局平均池化层操作 ([cecfe69](https://github.com/zjZSTU/PyNet/commit/cecfe69))
* **layer:** 添加softmax评分类实现 ([ab85f10](https://github.com/zjZSTU/PyNet/commit/ab85f10))
* **layers:** 添加动量更新 ([b8e1e60](https://github.com/zjZSTU/PyNet/commit/b8e1e60))
* **LeNet-5:** 取出卷积层失活操作 ([57b8e5e](https://github.com/zjZSTU/PyNet/commit/57b8e5e))
* **nets:** LeNet-5模型添加动量更新功能 ([b9cac97](https://github.com/zjZSTU/PyNet/commit/b9cac97))
* **NIN:** 实现NIN类 ([f06b1b6](https://github.com/zjZSTU/PyNet/commit/f06b1b6))
* **nn:** 实现2层神经网络类和LeNet-5 ([a846c20](https://github.com/zjZSTU/PyNet/commit/a846c20))
* **nn:** 实现conv class、maxpool class、fc class、relu class和CrossEntropyLoss class ([dba4057](https://github.com/zjZSTU/PyNet/commit/dba4057))
* **nn:** 实现Nesterov加速梯度 ([580d0a6](https://github.com/zjZSTU/PyNet/commit/580d0a6))
* **nn:** 实现三层神经网络的随机失活，分离前向训练和预测 ([60f6f65](https://github.com/zjZSTU/PyNet/commit/60f6f65))
* **refactor:** 重构网络类和模型设计，参考Pytorch/models结构，使用面向对象方法 ([5f0ecec](https://github.com/zjZSTU/PyNet/commit/5f0ecec))
* **src:** 网络测试 ([05fb51d](https://github.com/zjZSTU/PyNet/commit/05fb51d))



