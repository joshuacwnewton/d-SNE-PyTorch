"""
    Utility functions too specific for inclusion in other modules. These
    functions should possibly be moved as per the following article:
    https://breadcrumbscollector.tech/stop-naming-your-python-modules-utils/

    Various utility functions have been adapted from code contained
    in pytorch-template repo. That code has been heavily modified, but
    credit is due for providing a foundation for this code. See this
    link for further information:

    https://github.com/victoresque/pytorch-template
"""

# Stdlib imports
import os
from pathlib import Path
import glob
import json
import configparser
import argparse
from collections import OrderedDict

# Third-party imports
import numpy as np
import torch


def set_random_seeds(seed):
    """Set random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def prepare_device(n_gpu_requested, logger):
    """Determine suitable device (GPU/CPU) based on availability."""
    n_gpu_available = torch.cuda.device_count()
    if n_gpu_requested > n_gpu_available:
        logger.warning(f"Warning: {n_gpu_requested} GPUs requested "
                       f"but only {n_gpu_available} GPUs available."
                       f"Training on minimum GPUs. (Note: 0 => CPU)")
        n_gpu_used = n_gpu_available
    else:
        n_gpu_used = n_gpu_requested

    device = torch.device('cuda:0' if n_gpu_requested > 0 else 'cpu')

    return device, n_gpu_used


def ensure_dir(dirname):
    """Ensure that requested directory exists."""
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    """Read JSON-formatted file into an OrderedDict object."""
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """Write content to file using JSON formatting."""
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_most_recent_file(src_dir, fname, recursive=True):
    """Find most recent file in `save_dir` with name `fname`."""
    glob_ptn = f"{src_dir}/**/{fname}" if recursive else f"{src_dir}/{fname}"
    list_of_files = glob.glob(glob_ptn, recursive=recursive)

    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        raise FileNotFoundError(f"No files called {fname} in {src_dir}.")

    return latest_file


def parse_config(config_path, rem_args):
    """Parse config file, then override entries with any CLI arguments
    which match the config file keys."""
    # Read cfg file into ConfigParser object
    config = configparser.ConfigParser()
    config.read(config_path)

    # Convert ConfigParser object to dict of dicts
    config_dict = {k: dict(v) for k, v in dict(config).items()}

    # Convert '[Section][Key] = Value' mapping into '[Key] = Section' mapping
    section_dict = {}
    for section, item_dict in config_dict.items():
        for key, value in item_dict.items():
            if key.lower() in section_dict:
                raise Warning(f"Warning: Duplicate key '{value}' in config"
                              f" file. If passed using CLI args, the values"
                              f" for all matching keys will be overridden.")
            section_dict[key] = section

    # Create an argument for each key in the config file
    parser = argparse.ArgumentParser()
    for key in section_dict.keys():
        parser.add_argument(f"--{key}")

    # Parse args to check if any of the config keys were specified
    args = parser.parse_args(rem_args)
    for arg_name, arg_val in vars(args).items():
        if arg_val:
            # Override the cfg file if a matching argument was passed
            section_name = section_dict[arg_name]
            config[section_name][arg_name] = arg_val

    return config
