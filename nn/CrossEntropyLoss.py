# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:12
# @Author  : zj


from .pool2row import *

__all__ = ['Conv2d']


class CrossEntropyLoss(object):

    def __init__(self):
        self.probs = None
        self.labels = None

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        # scores.shape == [N, grad_out]
        # labels.shape == [N]
        assert len(scores.shape) == 2
        assert len(labels.shape) == 1
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        self.probs = expscores / np.sum(expscores, axis=1, keepdims=True)
        self.labels = labels.copy()

        N = labels.shape[0]
        correct_probs = self.probs[range(N), labels]
        loss = -1.0 / N * np.sum(np.log(correct_probs))
        return loss

    def backward(self):
        # grad_out = probs - Y
        grad_out = self.probs
        N = self.labels.shape[0]

        grad_out[range(N), self.labels] -= 1
        return grad_out

    def get_probs(self):
        return self.probs
