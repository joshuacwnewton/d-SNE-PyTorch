# d-SNE PyTorch

A PyTorch port of the MXNet (Gluon) implementation for `d-SNE: Domain Adaptation using Stochastic Neighbourhood Embedding` [(Xu et al., 2019)](https://arxiv.org/abs/1905.12775). This port was created by Joshua Newton to fulfill the requirements of the course project for _**CSC 586B**: Deep Learning for Computer Vision_ taught at the University of Victoria in BC, Canada.

**Note:** There are two original MXNet implementations: 

1. [October 2019](https://github.com/ShownX/d-SNE), a version originally created for the publication's CVPR submission.
2. [March 2020](https://github.com/ShownX/d-SNE), a refactored version created by author in response to GitHub issues ([1](https://github.com/aws-samples/d-SNE/issues/13), [2](https://github.com/aws-samples/d-SNE/issues/7)) which suggested that the October 2019 implementation was not functional. 

This port is primarily based off of the March 2020 implementation. 

## Usage

At time of writing, only the MNIST -> MNIST-M (supervised, 10 target images per class) experiment has been replicated. To run this experiment, use the following command: 

```
python3 -m dsne_pytorch configs/mt-mm.cfg --train --test
```

## Future work

Below is a list of the remaining functionality described by the source publication that has yet to be implemented. 

* VGG-16 and ResNet101 models. Used to compare performance with other state-of-the-art domain adaptation publications, including CCSA [(Motiian et al., 2017)](https://arxiv.org/abs/1709.10190) and FADA [(Motiian et al., 2017)](https://arxiv.org/abs/1711.02536).
* Dataset packing for various digits datasets ([USPS](https://www.kaggle.com/bistaumanga/usps-dataset), [SVHN](http://ufldl.stanford.edu/housenumbers/)), Office31 datasets ([Webcam, Amazon, DSLR](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)), and VisaDA-C 2017 datasets ([real, synthetic](https://ai.bu.edu/visda-2017/)). 
* Semi-supervised extension to allow use of unlabeled images.

Below is a list of experiments described by the source publication which have which have yet to be replicated.
* MNIST -> USPS (supervised, target images per class varying between 0 and 7)
* Various digits dataset pairs, including MNIST -> USPS, USPS -> MNIST, MNIST -> SVHN, and SVHN -> MNIST.
* Domain generalization tests to verify performance on source domain dataset.
* Various Office31 dataset pairs, including all combinations of Webcam, Amazon, and DSLR datasets.
* VisDA-C 2017 dataset, in accordance with challenge guidelines.

Lastly, it could be beneficial to further explore hyperparameter tuning. The default set of hyperparameters provided with the March 2020 MXNet implementation appear to have significant flaws.
* Loss/evolution curves are flat during the initial training iterations.
* Overfitting occurs before accuracy can reach the performance stated by the publication for the MNIST -> MNIST-M experiment. _(I achieved 81% compared to the 87% stated by the publication.)_

If time permits, this repository will be updated with further progress.

## Resources used in this project

Below are attributions for guides, tutorials, templates, and miscellaneous repositories that were consulted when implementing this project. 

* Package structure: https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
* PyTorch project structure: https://github.com/victoresque/pytorch-template
