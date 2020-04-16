"""
    Functions for calculating evaluation metrics.

    As of 2020-04-08, PyTorch does not provide metrics in its core
    package. This file provides relevant metric calculations. However,
    other commonly used third-party packages include scikit-learn and
    Ignite. This is discussed further in an open PyTorch feature
    request, which suggests that metrics may become part of PyTorch Core
    in an upcoming release.

    https://github.com/pytorch/pytorch/issues/22439

    Additionally, accuracy and top_k_acc funcs taken from code contained
    in pytorch-template repo. See this link for further information:

    https://github.com/victoresque/pytorch-template
"""

# Third-party imports
import torch
import pandas as pd
from numpy import inf


def accuracy(output, target):
    """Compute accuracy from class scores and target labels."""
    with torch.no_grad():
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, dim=0)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """Compute Top K accuracy from class scores and target labels."""
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


class MetricTracker:
    """Tracker of evaluation metrics for model selection.

    Attributes
    ----------
    metrics : str or list of str
        Names of metric functions. Must be present within this module's
        namespace to be used.
    best_metric : str
        Name of metric for evaluating whether a model has improved.
    best_mode : str
        'min' or 'max'. Indicates what improvement is for `best_metric`.
    best_val : float
        The best recorded value for `best_metric`.
    best_flag : Boolean
        Indicator for whether the most recent update was best.
    early_stop_count : int
        Condition for number of non-improvement updates needed in a row
        to stop early.
    not_improved_count : int
        Current number of non-improvement updates in a row.
    _data : Pandas DataFrame
        Table for tracking metrics.
    writer : SummaryWriter
        Object for passing metric history to Tensorboard.
    """

    def __init__(self, metrics, best_metric, best_mode, early_stop=None,
                 logger=None, writer=None, name=""):
        for metric in metrics:
            if not (metric in globals().keys() or metric == 'loss'):
                raise ValueError(f"{metric} not an available metric function")
        if best_metric not in metrics:
            raise ValueError(f"{best_metric} not one of tracked metrics.")
        if best_mode.lower() not in ['min', 'max']:
            raise ValueError("Mode for 'best' metric must be 'min' or 'max'.")

        self.name = f"{name: <10}"
        self.metrics = metrics

        # Attributes for checking whether a model has improved
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_val = inf if best_mode == 'min' else -inf
        self.best_flag = False

        # Attributes for early stopping
        self.epoch = 0
        self.early_stop_count = early_stop
        self.not_improved_count = 0

        self._data = pd.DataFrame(index=metrics,
                                  columns=['total', 'counts', 'average'])
        self.logger = logger
        self.writer = writer

        self.reset()

    @property
    def summary(self):
        """Provide averages for all tracked metrics."""
        return dict(self._data.average)

    @property
    def stop_early(self):
        """bool : whether condition to stop early has been met."""
        self.check_if_improved()
        if self.early_stop_count:
            stop_early = (self.not_improved_count >= self.early_stop_count)
        else:
            stop_early = False

        return stop_early

    def check_if_improved(self):
        """Whether metric avg has improved over previous best avg."""
        new_avg = self.avg(self.best_metric)

        if new_avg == self.best_val:
            self.best_flag = False  # No change, do nothing
        elif ((self.best_mode == 'min' and new_avg < self.best_val) or
              (self.best_mode == 'max' and new_avg > self.best_val)):
            self.not_improved_count = 0
            self.best_flag = True
            self.best_val = new_avg
        else:
            self.not_improved_count += 1
            self.best_flag = False

        return self.best_flag

    def avg(self, key):
        """Compute average for a single metric."""
        return self._data.average[key]

    def update(self, y_pred, y, loss=0, n=1):
        """Update tracked metrics and/or loss values."""
        for metric in self.metrics:
            if metric == 'loss':
                value = loss
            else:
                value = globals()[metric](y_pred, y)

            self._data.total[metric] += value * n
            self._data.counts[metric] += n
            self._data.average[metric] = (self._data.total[metric] /
                                          self._data.counts[metric])

    def log_event(self):
        if self.writer:
            self.writer.set_step(self.epoch - 1)

        log = {'mode': self.name, 'epoch': self.epoch}
        log.update(self.summary)

        log_str = ""
        for key, value in log.items():
            if self.writer and not isinstance(value, str):
                self.writer.add_scalar(key, value)

            f_value = f"{value:.4f}" if isinstance(value, float) else value
            log_str += f"{str(key).capitalize()}: {f_value} "

        if self.logger:
            self.logger.info(log_str)

    def reset_epoch(self):
        self.epoch += 1
        self.reset()

    def reset(self):
        """Reset internal metric dataframe to 0."""
        for col in self._data.columns:
            self._data[col].values[:] = 0
