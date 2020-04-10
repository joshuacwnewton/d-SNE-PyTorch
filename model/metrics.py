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
import pandas as pd
from numpy import inf


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


class MetricTracker:
    def __init__(self, metrics, stop_count=None, stop_mode=None,
                 stop_metric=None, writer=None):
        for metric in metrics:
            if not (metric in globals().keys() or metric == 'loss'):
                raise ValueError(f"{metric} not an available function")
        self.metrics = metrics

        # Validate and set parameters for early stopping
        self.best = 0
        if stop_count or stop_mode or stop_metric:
            if not stop_count and stop_mode and stop_metric:
                raise ValueError("Stop_epoch, stop_mode, and stop_metric must"
                                 "all be provided to use early stopping.")
            if stop_mode.lower() not in ['min', 'max']:
                raise ValueError("Stop_mode must be 'min' or 'max'.")
            if stop_metric not in metrics:
                raise ValueError(f"{stop_metric} not one of passed metrics.")

            self.best = inf if stop_mode == 'min' else -inf
        self.stop_mode = stop_mode
        self.stop_metric = stop_metric
        self.stop_count = stop_count
        self.improved = False
        self.not_improved_count = 0

        self._data = pd.DataFrame(index=metrics,
                                  columns=['total', 'counts', 'average'])
        self.writer = writer
        self.reset()

    @property
    def early_stop(self):
        self.check_if_improved()
        if self.stop_count:
            early_stop = self.not_improved_count >= self.stop_count
        else:
            early_stop = False

        return early_stop

    def check_if_improved(self):
        new_avg = self.avg(self.stop_metric)

        if new_avg == self.best:
            pass  # No change, do nothing
        elif ((self.stop_mode == 'min' and new_avg < self.best) or
              (self.stop_mode == 'max' and new_avg > self.best)):
            self.not_improved_count = 0
            self.improved = True
        else:
            self.not_improved_count += 1
            self.improved = False

        return self.improved

    @property
    def summary(self):
        return dict(self._data.average)

    def avg(self, key):
        return self._data.average[key]

    def update(self, loss, y_pred, y, n=1):
        for metric in self.metrics:
            if metric == 'loss':
                value = loss
            else:
                value = globals()[metric](y_pred, y)

            if self.writer is not None:
                self.writer.add_scalar(metric, value)
            self._data.total[metric] += value * n
            self._data.counts[metric] += n
            self._data.average[metric] = (self._data.total[metric] /
                                          self._data.counts[metric])

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
