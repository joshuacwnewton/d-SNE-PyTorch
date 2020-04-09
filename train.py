# Stdlib imports
import collections

# Third-party imports
import torch
import numpy as np

# Local application imports
from data_loading.dataloaders import get_dsne_dataloaders
from model.networks import LeNetPlus
from model.loss import CombinedLoss

# pytorch-template imports
import argparse
from pytorch_template import loggers
from pytorch_template.parse_config import ConfigParser
from pytorch_template.trainer import Trainer
from model.metric import accuracy, top_k_acc
from torch.optim import SGD
# from torch.optim.lr_scheduler import

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    loggers.setup_logging(config['trainer']['save_dir'])
    logger = loggers.get_logger(config['name'])
    writer = loggers.TensorboardWriter(config.log_dir, logger)

    train_dataloader = get_dsne_dataloaders("data_loading/data/mnist.h5",
                                            "data_loading/data/mnist_m.h5")
    model = LeNetPlus()
    criterion = CombinedLoss()
    metrics = [accuracy, top_k_acc]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = SGD(trainable_params, lr=0.001, weight_decay=0.0001,
                    momentum=0.9)

    trainer = Trainer(model, criterion, metrics, optimizer, logger, writer,
                      config=config, data_loader=train_dataloader)
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config',
                      default="pytorch_template/config.json", type=str,
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
