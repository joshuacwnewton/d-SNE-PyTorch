### Introduction

Before using `dsne_pytorch`, datasets must first be downloaded manually, then packed into an HDF5 container. Instructions for each dataset are provided below.

### Datasets

##### MNIST

>The MNIST database of handwritten digits [...] has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. 
>
>It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

MNIST can be downloaded from [the website of Yann LeCun](http://yann.lecun.com/exdb/mnist/) (from which this description was also taken). 

Please ensure that the `/data` directory contains the following structure before running the packing script:

```
/data/mnist/train-images-idx3-ubyte.gz
/data/mnist/t10k-images-idx3-ubyte.gz
/data/mnist/train-labels-idx1-ubyte.gz
/data/mnist/t10k-labels-idx1-ubyte.gz
```

Running the script will produce a file called `/data/mnist.h5` which can then be used within `dsne_pytorch`.

##### MNIST-M

MNIST-M is a modified version of the MNIST dataset. A description of the modification procedures can be found within the `Unsupervised Domain Adaptation by Backpropagation` publication [(Ganin and Lempitsky, 2011)](http://sites.skoltech.ru/compvision/projects/grl/).

>In order to obtain the target domain (MNIST-M) we blend digits from the original set over patches randomly extracted from color photos from BSDS500 (Arbelaez et al., 2011).

MNIST-M can be downloaded from [the website of Yaroslav Ganin](http://yaroslav.ganin.net/). Please make sure to download the "unpacked version of MNIST-M".

Please ensure that the `/data` directory contains the following structure before running the packing script:

```
/data/mnist_m/mnist_m_train/
/data/mnist_m/mnist_m_test/
/data/mnist_m/mnist_m_train_labels.txt
/data/mnist_m/mnist_m_test_labels.txt
```

Running the script will produce a file called `/data/mnist_m.h5` which can then be used within `dsne_pytorch`.


### HDF5 Packing

Once downloaded, each dataset can be packed using the `pack_data_hdf5.py` script found in `/dsne_pytorch/data_loading`. Example usage is provided below.

```
python3 pack_data_hdf5 mnist mnist-m
```
