
# PyNet

![](./imgs/logo.png)

[![Documentation Status](https://readthedocs.org/projects/zj-pynet/badge/?version=latest)](https://zj-pynet.readthedocs.io/zh_CN/latest/?badge=latest) [![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org) [![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> 基于Numpy的深度学习库

基于`Numpy`的深度学习实现，模块化设计保证模型的轻松实现，适用于深度学习初级研究人员的入门

[使用示例](https://zj-pynet.readthedocs.io/zh_CN/latest/%E7%A4%BA%E4%BE%8B/)

***这个项目不再继续了。最开始的想法很简单，就是要深入学习卷积神经网络。不过随着项目的深入会发现确实有很多的约束。不管这么样，对于刚刚开始入门深度学习的童鞋们来说，看看这里的源码还是很有帮助的***

## 内容列表

- [PyNet](#pynet)
  - [内容列表](#内容列表)
  - [背景](#背景)
  - [徽章](#徽章)
  - [安装](#安装)
  - [用法](#用法)
  - [版本更新日志](#版本更新日志)
  - [待办事项](#待办事项)
  - [主要维护人员](#主要维护人员)
  - [致谢](#致谢)
  - [参与贡献方式](#参与贡献方式)
  - [许可证](#许可证)

## 背景

系统性的学习卷积神经网络也快半年了，使用`pytorch`等库不能很好的深入理解实现，所以打算从头完成一个深度学习框架。最开始的实现会参考`cs231n`的作业，之后会以计算图的方式实现。希望这个项目能够切实提高自己的编程能力，同时也能够帮助到其他人

## 徽章

如果你使用了`PyNet`，请添加以下徽章

[![pynet](https://img.shields.io/badge/pynet-ok-brightgreen)](https://github.com/zjZSTU/PyNet)

Markdown格式代码如下：

```
[![pynet](https://img.shields.io/badge/pynet-ok-brightgreen)](https://github.com/zjZSTU/PyNet)
```

## 安装

```
# 文档工具依赖
$ pip install -r requiremens.txt
# PyNet库依赖
$ cd pynet
$ pip install -r requirements.txt
```

## 用法

有两种文档使用方式

1. 在线浏览文档：[PyNet](https://zj-pynet.readthedocs.io/zh_CN/latest/)

2. 本地浏览文档，实现如下：

    ```
    $ git clone https://zj-pynet.readthedocs.io/zh_CN/latest/
    $ cd PyNet
    $ mkdocs serve
    ```
   启动本地服务器后即可登录浏览器`localhost:8000`

## 版本更新日志

请参阅仓库中的[CHANGELOG](./CHANGELOG.md)

## 待办事项

* 计算图实现

## 主要维护人员

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## 致谢

* [cs231n](http://cs231n.github.io/)
* [PyTorch](https://pytorch.org/)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjZSTU/PyNet/issues)或提交合并请求。

注意:

* `git`提交请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* 如果进行版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* 如果修改README，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2019 zjZSTU