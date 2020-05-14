"""
    Contains agent classes which facilitate training/testing procedures.

    Trainer class adapted from code contained in pytorch-template repo.
    That code has been heavily modified, but credit is due for providing
    a foundation for this code. See this link for further information:

    https://github.com/victoresque/pytorch-template
"""

# Stdlib imports
from pathlib import Path

# Third-party imports
import numpy as np
import torch

# Local application imports
from dsne_pytorch.data_loading.data_classes import InfLoader


class Trainer:
    """
    Class that facilitates communication between ML objects (dataset,
    model, loss, optimizer, criterion) to train neural networks.
    """
    def __init__(self, train_loader, valid_loader, model, criterion, optimizer,
                 train_tracker, valid_tracker, logger, device, epochs,
                 save_period, save_dir, len_epoch=None, resume=None):
        # Set logging functions
        self.logger = logger
        self.log_step = int(np.sqrt(train_loader.batch_size))

        # Set device configuration (CPU/GPU) then move model to device
        self.device = device
        self.model = model.to(self.device)

        # Set remaining core objects needed for training
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_tracker = train_tracker
        self.valid_tracker = valid_tracker

        # Set config for duration of training
        self.start_epoch = 1
        self.epochs = epochs
        if len_epoch is None:
            self.len_epoch = len(self.train_loader)
        else:
            # Treat loader as repeating stream of data
            self.train_loader = InfLoader(self.train_loader)
            self.len_epoch = len_epoch

        # Set config for saving/reloading checkpoints
        self.save_period = save_period
        self.checkpoint_dir = Path(save_dir) / "ckpt"
        self.checkpoint_dir.mkdir()
        if resume is not None:
            self._resume_checkpoint(resume)

        self.logger.info("Trainer fully initialized. Starting training now...")

    def _save_checkpoint(self, epoch, save_best=False):
        """Save a PyTorch model checkpoint to a file."""
        state = {
            'arch_type': type(self.model).__name__,
            'optim_type': type(self.optimizer).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.valid_tracker.best_val,
        }
        filenames = []
        if epoch % self.save_period == 0:
            filenames.append(self.checkpoint_dir /
                             f'checkpoint-epoch{epoch}.pth')
        if save_best:
            filenames.append(self.checkpoint_dir / 'model_best.pth')

        for fn in filenames:
            self.logger.info(f"Saving model (epoch {epoch}) as '{fn.name}'...")
            torch.save(state, fn)

    def _resume_checkpoint(self, resume_path):
        """Resume progress from a previously saved checkpoint."""
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)

        # Warn if type names don't match
        if checkpoint['arch_type'] != type(self.model).__name__:  # noqa
            self.logger.warning(
                "Warning: Architecture type passed to Trainer is different"
                " from that of checkpoint. This may yield an exception while"
                " state_dict is being loaded."
            )
        if checkpoint['optim_type'] != type(self.optimizer).__name__:  # noqa
            self.logger.warning(
                "Warning: Optimizer type passed to Trainer is different"
                " from that of checkpoint. This may yield an exception while"
                " state_dict is being loaded."
            )

        # Load relevant values from checkpoint dict
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_tracker.epoch = self.start_epoch
        self.valid_tracker.epoch = self.start_epoch
        self.valid_tracker.best_val = checkpoint['best_metric']

        self.logger.info(f"Checkpoint loaded. Resuming training from "
                         f"epoch {self.start_epoch}...")

    def train(self):
        """Iterate over training epochs and save best models."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.debug(f"=============== Epoch {epoch}/{self.epochs} "
                              f"In Progress ==============")
            self._train_epoch()
            self.train_tracker.log_event()

            best = self._valid_epoch()
            self.valid_tracker.log_event()

            self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self):
        """Train model using batches from training set."""
        self.model.train()
        self.train_tracker.reset_epoch()

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X = {k: v.to(self.device) for k, v in X.items()}  # Send X to GPU
            y = {k: v.to(self.device) for k, v in y.items()}  # Send y to GPU

            # Repeat train step for both target and source datasets
            for train_name in X.keys():
                self.optimizer.zero_grad()

                ft, y_pred = {}, {}
                for name in X.keys():
                    ft[name], y_pred[name] = self.model(X[name])

                loss = self.criterion(ft, y_pred, y, train_name)
                loss.backward()

                self.optimizer.step()

            self.train_tracker.update(y_pred['src'], y['src'],
                                      loss=loss.item(),
                                      n=self.train_loader.batch_size)

            if batch_idx == self.len_epoch:
                break

    def _valid_epoch(self):
        """Evaluate model using batches from validation set."""
        self.model.eval()
        self.valid_tracker.reset_epoch()

        with torch.no_grad():
            for X, y in self.valid_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                features, output = self.model(X)
                self.valid_tracker.update(output, y)

        return self.valid_tracker.check_if_improved()

    def _progress(self, batch_idx):
        """Generate string representing elapsed training progress."""
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Tester:
    """
    Class that facilitates communication between ML objects (dataset,
    model, metric tracker, logger) to evaluate neural networks.
    """
    def __init__(self, data_loader, model, ckpt_path, metric_tracker,
                 device, logger):
        self.device = device

        self.data_loader = data_loader

        self.ckpt_path = ckpt_path
        self.ckpt_dict = torch.load(ckpt_path)
        model.load_state_dict(self.ckpt_dict["state_dict"])
        self.model = model.to(self.device).eval()

        self.metric_tracker = metric_tracker
        self.metric_tracker.reset()

        self.logger = logger

    def test(self):
        """Iterate over test dataset and record evaluation metrics."""
        with torch.no_grad():
            for X, y in self.data_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                features, output = self.model(X)
                self.metric_tracker.update(output, y)

        log = {'mode': 'test'}
        log.update(self.metric_tracker.summary)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))