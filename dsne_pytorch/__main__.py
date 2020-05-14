"""
    PyTorch port of the MXNet (Gluon) implementation for the 2019 CVPR paper
    'd-SNE: Domain Adaptation using Stochastic Neighbourhood Embedding'.

    Author: Joshua Newton (V00826800)
    Email: jnewt@uvic.ca
"""


# Stdlib imports
import os
import argparse
from datetime import datetime
from pathlib import Path

# Third-party imports
from torch.optim import SGD

# Local application imports
from dsne_pytorch import agents, loggers, utils
from dsne_pytorch.data_loading.data_classes import init_dataloaders
from dsne_pytorch.model.networks import LeNetPlus
from dsne_pytorch.model.loss import CombinedLoss
from dsne_pytorch.model.metrics import MetricTracker


def main():
    # Ensure current working directory is the package root (dsne_pytorch/)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # First, parse args unrelated to the d-SNE config, then pass the remaining
    # args to be parsed along with the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="Path to test configuration file")
    parser.add_argument('--train', action="store_true", help="Training flag")
    parser.add_argument('--test', action="store_true", help="Testing flag")
    args, rem_argv = parser.parse_known_args()
    config = utils.parse_config(args.config_path, rem_argv)

    # Generate path for unique experiment directory using current time
    output_dir = Path(config["General"]["output_dir"])
    test_type = config["General"]["test_name"]
    test_id = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    config["Training"]["output_dir"] = str(output_dir / test_type / test_id)

    # Ensure that any results are reproducible
    utils.set_random_seeds(123)

    # Create instances of ML classes using specified configuration
    objs = init_objects(config)

    # Ensure that requested number of GPUs is available
    device, n_gpu = utils.prepare_device(config["General"].getint("n_gpu"),
                                         objs["logger"])

    if args.train:
        trainer = agents.Trainer(
            train_loader=objs["train_loader"],
            valid_loader=objs["valid_loader"],
            model=objs["model"],
            criterion=objs["criterion"],
            optimizer=objs["optimizer"],
            train_tracker=objs["train_tracker"],
            valid_tracker=objs["valid_tracker"],
            logger=objs["logger"],
            device=device,
            epochs=config["Training"].getint("epochs"),
            len_epoch=config["Training"].getint("len_epoch"),
            save_period=config["Training"].getint("save_period"),
            save_dir=config["Training"]["output_dir"],
            resume=config["Training"].get("resume")
        )
        trainer.train()

    if args.test:
        # Load backup checkpoint if no path is specified
        if "ckpt" not in config["Testing"]:
            config["Testing"]["ckpt"] = utils.get_most_recent_file(
                Path(config["Training"]["output_dir"]).parent, "model_best.pth"
            )

        tester = agents.Tester(
            data_loader=objs["test_loader"],
            model=objs["model"],
            metric_tracker=objs["valid_tracker"],
            logger=objs["logger"],
            device=device,
            ckpt_path=config["Testing"]["ckpt"]
        )
        tester.test()


def init_objects(config):
    """Initialize objects which perform various ML duties."""
    objs = {}

    loggers.setup_logging(save_dir=config['Training']['output_dir'])
    objs["logger"] = loggers.get_logger(name=config['General']['test_name'])
    objs["train_writer"] = loggers.TensorboardWriter(
        log_dir=config['Training']['output_dir'],
        logger=objs["logger"],
        subfolder="train"
    )
    objs["valid_writer"] = loggers.TensorboardWriter(
        log_dir=config['Training']['output_dir'],
        logger=objs["logger"],
        subfolder="valid"
    )

    objs["train_loader"], objs["valid_loader"], objs["test_loader"] \
        = init_dataloaders(
            src_path=config['Datasets']['src_path'],
            tgt_path=config['Datasets']['tgt_path'],
            src_num=config['Datasets'].getint('src_num'),
            tgt_num=config['Datasets'].getint('tgt_num'),
            sample_ratio=config['Datasets'].getint('sample_ratio'),
            resize_dim=config['Datasets'].getint('resize_dim'),
            batch_size=config['Datasets'].getint('batch_size'),
            shuffle=config['Datasets'].getboolean('shuffle')
        )

    objs["model"] = LeNetPlus(
        input_dim=config['Model'].getint('input_dim'),
        classes=config['Model'].getint('classes'),
        feature_size=config['Model'].getint('feature_size'),
        dropout=config['Model'].getfloat('dropout')
    )

    objs["train_tracker"] = MetricTracker(
        metrics=config["Metrics"]["funcs"].split(),
        best_metric=config["Metrics"]["best_metric"],
        best_mode=config["Metrics"]["best_mode"],
        logger=objs["logger"],
        writer=objs["train_writer"],
        name="Training"
    )
    objs["valid_tracker"] = MetricTracker(
        metrics=config["Metrics"]["funcs"].split(),
        best_metric=config["Metrics"]["best_metric"],
        best_mode=config["Metrics"]["best_mode"],
        logger=objs["logger"],
        writer=objs["valid_writer"],
        name="Validation"
    )

    objs["criterion"] = CombinedLoss(
        margin=config['Loss'].getfloat('margin'),
        alpha=config['Loss'].getfloat('alpha')
    )

    trainable_params = filter(lambda p: p.requires_grad,
                              objs["model"].parameters())
    objs["optimizer"] = SGD(
        params=trainable_params,
        lr=config['Optimizer'].getfloat('learning_rate'),
        weight_decay=config['Optimizer'].getfloat('weight_decay'),
        momentum=config['Optimizer'].getfloat('momentum')
    )

    return objs


if __name__ == "__main__":
    main()

