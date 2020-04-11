# Stdlib imports
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
from dsne_pytorch.agents import DSNETrainer, Tester
from dsne_pytorch.utils import (fix_random_seeds, prepare_device,
                                get_latest_model)


def main(args, config):
    fix_random_seeds(123)

    loggers.setup_logging(save_dir=config['General']['test_dir'])
    logger = loggers.get_logger(name=config['General']['test_name'])
    writer = loggers.TensorboardWriter(log_dir=config['General']['test_dir'],
                                       logger=logger)

    device, n_gpu = prepare_device(config["General"].getint("n_gpu"), logger)

    train_dataloader, test_dataloader = get_dsne_dataloaders(
            src_path=config['Datasets']['src_path'],
            tgt_path=config['Datasets']['tgt_path'],
             src_num=config['Datasets'].getint('src_num'),
             tgt_num=config['Datasets'].getint('tgt_num'),
        sample_ratio=config['Datasets'].getint('sample_ratio'),
           image_dim=config['Datasets'].getint('image_dim'),
          batch_size=config['Datasets'].getint('batch_size'),
             shuffle=config['Datasets'].getboolean('shuffle')
    )

    model = LeNetPlus(
        input_dim=config['Datasets'].getint('image_dim'),
        classes=config['Model'].getint('classes'),
        feature_size=config['Model'].getint('feature_size'),
        dropout=config['Model'].getfloat('dropout')
    )

    metric_tracker = MetricTracker(
        metrics=config["Metrics"]["funcs"].split(),
        best_metric=config["Metrics"]["best_metric"],
        best_mode=config["Metrics"]["best_mode"]
    )

    if args.train:
        criterion = CombinedLoss(
            margin=config['Loss'].getfloat('margin'),
             alpha=config['Loss'].getfloat('alpha')
        )

        trainable_params = filter(lambda p: p.requires_grad,
                                  model.parameters())
        optimizer = SGD(
            trainable_params,
                      lr=config['Optimizer'].getfloat('learning_rate'),
            weight_decay=config['Optimizer'].getfloat('weight_decay'),
                momentum=config['Optimizer'].getfloat('momentum')
        )

        trainer = DSNETrainer(
            train_dataloader, model, criterion, optimizer,
            metric_tracker, logger, writer, device,
                 epochs=config["Training"].getint("epochs"),
              len_epoch=config["Training"].getint("len_epoch"),
            save_period=config["Training"].getint("save_period"),
               save_dir=config["General"]["test_dir"],
                 resume=config["Training"].get("resume")
        )
        trainer.train()

    if args.test:
        ckpt_path = config["Testing"].get(
            "ckpt",  # Or, get latest model if "ckpt" not in config
            get_latest_model(config["General"]["test_type_dir"],
                             "model_best.pth")
        )

        tester = Tester(test_dataloader, model, ckpt_path, metric_tracker,
                        device, logger)
        tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="Path to test configuration file")
    parser.add_argument('--train', action="store_true", help="Training flag")
    parser.add_argument('--test', action="store_true", help="Testing flag")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    # Generate unique test directories on the fly
    output_dir = Path(config["General"]["output_dir"])
    test_type = config["General"]["test_name"]
    test_id = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    config["General"]["test_type_dir"] = str(output_dir / test_type)
    config["General"]["test_dir"] = str(output_dir / test_type / test_id)

    main(args, config)

