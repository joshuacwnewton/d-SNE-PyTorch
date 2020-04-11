import os
import json
from pathlib import Path
from collections import OrderedDict
import glob

import numpy as np
import torch


def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def prepare_device(n_gpu_requested, logger):
    """
    setup GPU device if available, move model into configured device
    """
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
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_latest_model(save_dir, model_name):
    list_of_files = glob.glob(f"{save_dir}/**/{model_name}", recursive=True)
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        raise FileNotFoundError(f"No files called {model_name} in {save_dir}.")

    return latest_file
