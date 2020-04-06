# Stdlib imports
import logging

# Third-party imports
import torch
import numpy as np

# Local application imports
from data_loading.dataloaders import get_dsne_dataloaders
from model.networks import LeNetPlus
from model.loss import CombinedLoss

# pytorch-template imports
import argparse
import collections
from parse_config import ConfigParser
from trainer.trainer import Trainer
from logger import setup_logging
from model.metric import accuracy, top_k_acc
from torch.optim import SGD
# from torch.optim.lr_scheduler import

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation()
    train_dataloader = get_dsne_dataloaders("data_loading/data/mnist.h5",
                                            "X_tr", "y_tr",
                                            "data_loading/data/mnist_m.h5",
                                            "X_tr", "y_tr")

    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch)
    model = LeNetPlus()
    logger.info(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    criterion = CombinedLoss(margin=1.0, fn=False)
    metrics = [accuracy, top_k_acc]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # using DSNE params
    optimizer = SGD(trainable_params,
                    lr=0.001, weight_decay=0.0001, momentum=0.9)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_dataloader)
    trainer.train()

    test = None


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
