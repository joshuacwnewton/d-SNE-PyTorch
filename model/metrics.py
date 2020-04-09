"""Functions for calculating evaluation metrics.

As of 2020-04-08, PyTorch does not provide metrics in its core package.
This file provides relevant metric calculations. However, other
commonly used third-party packages include scikit-learn and Ignite.
This is discussed further in an open PyTorch feature request, which
suggests that metrics may become part of PyTorch Core in an upcoming
release.

    https://github.com/pytorch/pytorch/issues/22439
"""

import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
