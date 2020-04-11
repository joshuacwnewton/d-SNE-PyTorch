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
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, dim=0)
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
    def __init__(self, metrics, best_metric, best_mode, early_stop=None,
                 writer=None):
        for metric in metrics:
            if not (metric in globals().keys() or metric == 'loss'):
                raise ValueError(f"{metric} not an available metric function")
        if best_metric not in metrics:
            raise ValueError(f"{best_metric} not one of tracked metrics.")
        if best_mode.lower() not in ['min', 'max']:
            raise ValueError("Mode for 'best' metric must be 'min' or 'max'.")

        self.metrics = metrics
        self._data = pd.DataFrame(index=metrics,
                                  columns=['total', 'counts', 'average'])
        self.writer = writer

        # Attributes for checking whether a model has improved
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_val = inf if best_mode == 'min' else -inf
        self.best_flag = False

        # Attributes for early stopping
        self.early_stop_count = early_stop
        self.not_improved_count = 0

        self.reset()

    @property
    def stop_early(self):
        self.check_if_improved()
        if self.early_stop_count:
            stop_early = (self.not_improved_count >= self.early_stop_count)
        else:
            stop_early = False

        return stop_early

    def check_if_improved(self):
        new_avg = self.avg(self.best_metric)

        if new_avg == self.best_val:
            pass  # No change, do nothing
        elif ((self.best_mode == 'min' and new_avg < self.best_val) or
              (self.best_mode == 'max' and new_avg > self.best_val)):
            self.not_improved_count = 0
            self.best_flag = True
        else:
            self.not_improved_count += 1
            self.best_flag = False

        return self.best_flag

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
