from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid

from pytorch_template.utils import inf_loop


class DSNETrainer:
    """
    Base class for all trainers
    """
    def __init__(self, data_loader, model, criterion, optimizer,
                 metric_tracker, logger, writer, n_gpu, epochs, save_period,
                 save_dir, len_epoch=None, resume=None):
        # Set logging functions
        self.logger = logger
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.writer = writer

        # Set device configuration (CPU/GPU) then move model to device
        self.n_gpu_used = self._prepare_device(n_gpu)
        self.model = model.to(self.device)
        if self.n_gpu_used > 1:
            self.model = torch.nn.DataParallel(
                model, device_ids=list(range(self.n_gpu_used))
            )

        # Set remaining core objects needed for training
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_tracker = metric_tracker

        # Set config for duration of training
        self.start_epoch = 1
        self.epochs = epochs
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        # Set config for saving/reloading checkpoints
        self.save_period = save_period
        self.checkpoint_dir = Path(save_dir) / "ckpt"
        self.checkpoint_dir.mkdir()
        if resume is not None:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_requested):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu_available = torch.cuda.device_count()
        if n_gpu_requested > n_gpu_available:
            self.logger.warning(f"Warning: {n_gpu_requested} GPUs requested "
                                f"but only {n_gpu_available} GPUs available."
                                f"Training on minimum GPUs. (Note: 0 => CPU)")
            n_gpu_used = n_gpu_available
        else:
            n_gpu_used = n_gpu_requested

        self.device = torch.device('cuda:0' if n_gpu_requested > 0 else 'cpu')

        return n_gpu_used

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'arch_type': type(self.model).__name__,
            'optim_type': type(self.optimizer).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.metric_tracker.best_val,
        }

        filenames = [self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth']
        if save_best:
            filenames.append(self.checkpoint_dir / 'model_best.pth')

        for fn in filenames:
            self.logger.info(f"Saving epoch {epoch} using ckpt name '{fn}'...")
            torch.save(state, fn)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
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
        self.metric_tracker.best_val = checkpoint['best_metric']

        self.logger.info(f"Checkpoint loaded. Resuming training from "
                         f"epoch {self.start_epoch}...")

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            best = self._train_epoch(epoch)

            # Print epoch summary to logger
            log = {'epoch': epoch}
            log.update(self.metric_tracker.summary)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.metric_tracker.reset()

        for batch_idx, (X, y) in enumerate(self.data_loader):
            # TODO: Keep train step for DSNETrainer, move train epoch to Base
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

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.metric_tracker.update(loss.item(), y_pred['src'], y['src'])

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f"Train Epoch: {epoch} {self._progress(batch_idx)}"
                    f" Loss: {loss.item():.6f}"
                    f" Accuracy: {self.metric_tracker.avg('accuracy'):.4f}")
                self.writer.add_image('source', make_grid(X['src'].cpu(),
                                                          nrow=8,
                                                          normalize=True))
                self.writer.add_image('target', make_grid(X['tgt'].cpu(),
                                                          nrow=8,
                                                          normalize=True))
            if batch_idx == self.len_epoch:
                break

        return self.metric_tracker.check_if_improved()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
