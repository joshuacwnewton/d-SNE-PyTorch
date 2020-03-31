"""Standalone script for packing downloaded datasets into HDF5
container."""


from pathlib import Path
import gzip
import struct

import numpy as np


def mnist(root_path):
    # Set up paths for required directory and files
    mnist_path = Path(root_path)
    required_files = {"X_tr": "train-images-idx3-ubyte.gz",
                      "y_tr": "train-labels-idx1-ubyte.gz",
                      "X_te": "t10k-images-idx3-ubyte.gz",
                      "y_te": "t10k-labels-idx1-ubyte.gz"}

    # Check to see if required files exist
    assert mnist_path.exists()
    for file_path in required_files.values():
        assert (mnist_path / file_path).exists()

    # Load images and labels into dataset dictionary
    dataset = {}
    for name, file_path in required_files.items():
        with gzip.open(mnist_path / file_path, 'rb') as f:
            if name.startswith("X"):
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8)
                dataset[name] = data.reshape(-1, rows, cols)

            elif name.startswith("y"):
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.int8)
                dataset[name] = data

    return dataset


def mnist_m(root_path):
    pass


def svhn():
    pass


def usps():
    pass


def office31():
    pass


def visdac_2017():
    pass


def pack_dataset():
    pass


def main():
    # argparsing for which datasets you'd like to pack
    pass
