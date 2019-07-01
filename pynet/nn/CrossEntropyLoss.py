# -*- coding: utf-8 -*-

# @Time    : 19-6-20 下午3:12
# @Author  : zj


from .pool2row import *

__all__ = ['Conv2d']


class CrossEntropyLoss(object):

    def __call__(self, scores, labels):
        return self.forward(scores, labels)

    def forward(self, scores, labels):
        # scores.shape == [N, score]
        # labels.shape == [N]
        assert len(scores.shape) == 2
        assert len(labels.shape) == 1
        scores -= np.max(scores, axis=1, keepdims=True)
        expscores = np.exp(scores)
        probs = expscores / np.sum(expscores, axis=1, keepdims=True)

        N = labels.shape[0]
        correct_probs = probs[range(N), labels]
        loss = -1.0 / N * np.sum(np.log(correct_probs))
        return loss, probs

    def backward(self, probs, labels):
        assert len(probs.shape) == 2
        assert len(labels.shape) == 1
        grad_out = probs
        N = labels.shape[0]

        grad_out[range(N), labels] -= 1
        return grad_out
