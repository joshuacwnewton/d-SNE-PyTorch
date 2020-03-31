"""Standalone script for packing downloaded datasets into HDF5
container."""

import argparse
from pathlib import Path
import gzip
import struct

import numpy as np
import h5py


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


def pack_dataset(output_path, dataset):
    output_path = Path(output_path)
    if output_path.exists():
        renamed_path = f"{output_path.stem}_backup{output_path.suffix}"
        print(f"{output_path} already exists. Renaming to {renamed_path}.")
        output_path.rename(renamed_path)

    file = h5py.File(output_path, "w")

    for split_name in ["tr", "te"]:
        image_name = f"X_{split_name}"
        label_name = f"y_{split_name}"

        images = dataset[image_name]
        labels = dataset[label_name]

        file.create_dataset(image_name, np.shape(images), h5py.h5t.STD_U8BE,
                            data=images)
        file.create_dataset(label_name, np.shape(labels), h5py.h5t.STD_U8BE,
                            data=labels)

    file.close()


def main(requested_datasets):
    dataset_funcs = {"mt": mnist, "mnist": mnist,
                     "mm": mnist_m, "mnistm": mnist_m, "mnist-m": mnist_m,
                     "us": usps, "usps": usps,
                     "sn": svhn, "svhn": svhn}

    for dataset in requested_datasets:
        func = dataset_funcs[dataset]

        # Use function name as convention for directory/filenames
        dataset_path = Path("data")/func.__name__  # e.g. "data/mnist"
        output_filename = f"{func.__name__ }.h5"   # e.g. "mnist.h5"

        dataset = func(dataset_path)
        pack_dataset(output_filename, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    main(args.datasets)
