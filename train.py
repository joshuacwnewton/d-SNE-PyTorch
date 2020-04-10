# Stdlib imports
import argparse
import configparser
from datetime import datetime
from pathlib import Path

# Third-party imports
from torch.optim import SGD

# Local application imports
from data_loading.dataloaders import get_dsne_dataloaders
from model.networks import LeNetPlus
from model.loss import CombinedLoss
from model.metrics import MetricTracker
from pytorch_template import loggers
from pytorch_template.trainer import DSNETrainer
from pytorch_template.utils import fix_random_seeds


def main(config):
    fix_random_seeds(123)

    loggers.setup_logging(save_dir=config['General']['test_dir'])
    logger = loggers.get_logger(name=config['General']['test_name'])
    writer = loggers.TensorboardWriter(log_dir=config['General']['test_dir'],
                                       logger=logger)

    train_dataloader = get_dsne_dataloaders(
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
    criterion = CombinedLoss(
        margin=config['Loss'].getfloat('margin'),
         alpha=config['Loss'].getfloat('alpha')
    )
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = SGD(
        trainable_params,
                  lr=config['Optimizer'].getfloat('learning_rate'),
        weight_decay=config['Optimizer'].getfloat('weight_decay'),
            momentum=config['Optimizer'].getfloat('momentum')
    )
    metric_tracker = MetricTracker(config['Metrics']['funcs'].split())

    trainer = DSNETrainer(
        train_dataloader, model, criterion, optimizer,
        metric_tracker, logger, writer,
              n_gpu=config["Trainer"].getint("n_gpu"),
             epochs=config["Trainer"].getint("epochs"),
        save_period=config["Trainer"].getint("save_period"),
           save_dir=config["General"]["test_dir"]
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="Path to test configuration file")
    args = parser.parse_args()

    new_config = configparser.ConfigParser()
    new_config.read(args.config_path)

    # Generate unique test directory on the fly
    output_dir = Path(new_config["General"]["output_dir"])
    test_name = new_config["General"]["test_name"]
    test_id = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    new_config["General"]["test_dir"] = str(output_dir / test_name / test_id)

    main(new_config)
