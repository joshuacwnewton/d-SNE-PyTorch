import logging
import logging.config
from pathlib import Path
from pytorch_template.utils import read_json


def setup_logging(save_dir, default_level=logging.INFO,
                  log_config='pytorch_template/logger_config.json'):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
