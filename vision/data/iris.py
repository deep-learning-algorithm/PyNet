# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午4:25
# @Author  : zj

import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split


# iris_path = '/home/zj/data/iris-species/Iris.csv'


def load_iris(iris_path, shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(iris_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    return x_train, x_test, y_train, y_test
