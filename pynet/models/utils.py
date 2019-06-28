# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午2:16
# @Author  : zj


import pickle


def save_params(params, path='params.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(params, f, -1)


def load_params(path='params.pkl'):
    with open(path, 'rb') as f:
        param = pickle.load(f)
    return param


if __name__ == '__main__':
    a = {'a': 2341, 'b': 'adfadfa'}
    save_params(a)
    b = load_params()
    print(b)
