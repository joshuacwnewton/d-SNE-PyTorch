"""Standalone script for packing downloaded datasets into HDF5
container."""

import argparse
import os
from pathlib import Path
import gzip
import struct

import h5py
import numpy as np
import cv2


def mnist(root_path):
    # Set up paths for required directory and files
    mnist_path = Path(root_path)
    required_files = {"y_tr": "train-labels-idx1-ubyte.gz",
                      "X_tr": "train-images-idx3-ubyte.gz",
                      "y_te": "t10k-labels-idx1-ubyte.gz",
                      "X_te": "t10k-images-idx3-ubyte.gz"}

    # Check to see if required files exist
    assert mnist_path.exists()
    for file_path in required_files.values():
        assert (mnist_path / file_path).exists()

    # Load labels and images into dataset dictionary
    dataset = {}
    for name, file_path in required_files.items():
        with gzip.open(mnist_path / file_path, 'rb') as f:
            if name.startswith("y"):
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
                dataset[name] = data

            elif name.startswith("X"):
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8)
                dataset[name] = data.reshape(-1, rows, cols)

    return dataset


def mnist_m(root_path):
    # Set up paths for required directory and files
    mnist_m_path = Path(root_path)
    required_files = {"y_tr": "mnist_m_train_labels.txt",
                      "X_tr": "mnist_m_train",
                      "y_te": "mnist_m_test_labels.txt",
                      "X_te": "mnist_m_test"}

    # Check to see if required files exist
    assert mnist_m_path.exists()
    for file_path in required_files.values():
        assert (mnist_m_path / file_path).exists()

    # Load labels and images into dataset dictionary
    dataset = {}
    for name, file_path in required_files.items():
        if name.startswith('y'):
            with open(root_path/file_path, 'r') as f:
                paths, labels = zip(*[line.split() for line in f.readlines()])
                dataset[name] = np.array(labels, dtype=np.uint8)

        elif name.startswith("X"):
            images = [cv2.cvtColor(cv2.imread(str(root_path/file_path/path)),
                                   cv2.COLOR_BGR2RGB)
                      for path in paths]
            dataset[name] = np.array(images, dtype=np.uint8)

    return dataset


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
    # Ensure that current working directory is where pack_data_hdf5.py is
    os.chdir(Path(os.path.realpath(__file__)).parent)

    # Mapping from str arguments to function names
    dataset_funcs = {"mt": mnist, "mnist": mnist,
                     "mm": mnist_m, "mnistm": mnist_m, "mnist-m": mnist_m,
                     "us": usps, "usps": usps,
                     "sn": svhn, "svhn": svhn}

    for dataset in requested_datasets:
        try:
            func = dataset_funcs[dataset]

            # Use function name as convention for directory/filenames
            dataset_path = Path("data") / func.__name__  # e.g. "data/mnist"
            output_filename = f"{func.__name__}.h5"  # e.g. "mnist.h5"

            dataset = func(dataset_path)
            pack_dataset(output_filename, dataset)

        except KeyError:
            print(f'"{dataset}" does not map to valid dataset. Skipping it.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    main(args.datasets)
