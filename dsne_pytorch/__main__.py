"""
    Docstring for d-SNE PyTorch....
"""

# TODO: Write missing docstrings (main, agents, loggers, utils, metrics)
# TODO: Modify config to interpolate resize_dim from input_dim
# TODO: Switch to ExtendedInterpolation in configparsing
# TODO: Move "save_dir" from "General" category to "Training" category
# TODO: Add argparsing for cfg

# Stdlib imports
import os
import argparse
import configparser
from datetime import datetime
from pathlib import Path

# Third-party imports
from torch.optim import SGD

# Local application imports
from dsne_pytorch.data_loading.dataloaders import get_dsne_dataloaders
from dsne_pytorch.model.networks import LeNetPlus
from dsne_pytorch.model.loss import CombinedLoss
from dsne_pytorch.model.metrics import MetricTracker
from dsne_pytorch import loggers
from dsne_pytorch.agents import Trainer, Tester
from dsne_pytorch.utils import (fix_random_seeds, prepare_device,
                                get_latest_model)


def main():
    # Ensure current working directory is dsne_pytorch/
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="Path to test configuration file")
    parser.add_argument('--train', action="store_true", help="Training flag")
    parser.add_argument('--test', action="store_true", help="Testing flag")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    # Generate paths for unique experiment directories/files using current time
    output_dir = Path(config["General"]["output_dir"])
    test_type = config["General"]["test_name"]
    test_id = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    config["General"]["test_type_dir"] = str(output_dir / test_type)
    config["General"]["test_dir"] = str(output_dir / test_type / test_id)

    fix_random_seeds(123)
    objs = init_objects(config)
    device, n_gpu = prepare_device(config["General"].getint("n_gpu"),
                                   objs["logger"])

    if args.train:
        trainer = Trainer(
               data_loader=objs["train_loader"],
                     model=objs["model"],
                 criterion=objs["criterion"],
                 optimizer=objs["optimizer"],
            metric_tracker=objs["metric_tracker"],
                    logger=objs["logger"],
                    writer=objs["writer"],
                    device=device,
                    epochs=config["Training"].getint("epochs"),
                 len_epoch=config["Training"].getint("len_epoch"),
               save_period=config["Training"].getint("save_period"),
                  save_dir=config["General"]["test_dir"],
                    resume=config["Training"].get("resume")
        )
        trainer.train()

    if args.test:
        if "ckpt" not in config["Testing"]:
            config["Testing"]["ckpt"] = get_latest_model(
                config["General"]["test_type_dir"], "model_best.pth"
            )

        tester = Tester(
               data_loader=objs["test_loader"],
                     model=objs["model"],
            metric_tracker=objs["metric_tracker"],
                    logger=objs["logger"],
                    device=device,
                 ckpt_path=config["Testing"]["ckpt"]
        )
        tester.test()


def init_objects(config):
    """Initialize objects which perform various ML duties."""
    objs = {}

    loggers.setup_logging(save_dir=config['General']['test_dir'])
    objs["logger"] = loggers.get_logger(name=config['General']['test_name'])
    objs["writer"] = loggers.TensorboardWriter(
        log_dir=config['General']['test_dir'],
         logger=objs["logger"]
    )

    objs["train_loader"], objs["test_loader"] = get_dsne_dataloaders(
            src_path=config['Datasets']['src_path'],
            tgt_path=config['Datasets']['tgt_path'],
             src_num=config['Datasets'].getint('src_num'),
             tgt_num=config['Datasets'].getint('tgt_num'),
        sample_ratio=config['Datasets'].getint('sample_ratio'),
           image_dim=config['Datasets'].getint('image_dim'),
          batch_size=config['Datasets'].getint('batch_size'),
             shuffle=config['Datasets'].getboolean('shuffle')
    )

    objs["model"] = LeNetPlus(
           input_dim=config['Datasets'].getint('image_dim'),
             classes=config['Model'].getint('classes'),
        feature_size=config['Model'].getint('feature_size'),
             dropout=config['Model'].getfloat('dropout')
    )

    objs["metric_tracker"] = MetricTracker(
            metrics=config["Metrics"]["funcs"].split(),
        best_metric=config["Metrics"]["best_metric"],
          best_mode=config["Metrics"]["best_mode"]
    )

    objs["criterion"] = CombinedLoss(
        margin=config['Loss'].getfloat('margin'),
         alpha=config['Loss'].getfloat('alpha')
    )

    trainable_params = filter(lambda p: p.requires_grad,
                              objs["model"].parameters())
    objs["optimizer"] = SGD(
        trainable_params,
                  lr=config['Optimizer'].getfloat('learning_rate'),
        weight_decay=config['Optimizer'].getfloat('weight_decay'),
            momentum=config['Optimizer'].getfloat('momentum')
    )

    return objs


if __name__ == "__main__":
    main()

