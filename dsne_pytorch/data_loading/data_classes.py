"""
    Custom PyTorch data classes (Datasets, DataLoaders) used in d-SNE
    training and testing procedures.
"""

# Stdlib imports
from itertools import repeat

# Third-party imports
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, ToPILImage, ToTensor,
                                    Resize, Normalize)


class PairDataset(Dataset):
    """Combined source/target dataset for training using d-SNE.

    Attributes
    ----------
    X : dict of PyTorch Tensors (N, H, W, C)
        Images corresponding to samples of source/target datasets.
    y : dict of PyTorch Tensors (N, 1)
        Labels corresponding to samples of source/target datasets.
    intra_idxs : List of pairs of ints
        Indices for pairs of source/target samples w/ matching labels.
    inter_idxs : List of pairs of ints
        Indices for pairs of source/target samples w/o matching labels.
    full_idxs: List of pairs of ints
        Indexes for pairs of source/target samples.
    transform : Compose transform containing PyTorch transforms
        Pre-processing operations to apply to images when calling
        __getitem__.

    Methods
    -------
    __len__
        Reflect amount of available pairs of indices.
    __getitem__
        Get pair of source and target images/labels.

    Notes
    -----
    d-SNE trains networks using two datasets simultaneously. Of note,
    with d-SNE's training procedure, the loss calculation differs for
    intraclass pairs (y_src == y_tgt) versus interclass pairs
    (y_src != y_tgt).

    By pre-determining pairs of images using a paired dataset, the ratio
    of intraclass and interclass pairs can be controlled. This would be
    more difficult to manage if images were sampled separately from each
    dataset.
    """

    def __init__(self, src_path, tgt_path, src_num=-1, tgt_num=10,
                 sample_ratio=3, transform=()):
        """Initialize dataset by sampling subsets of source/target.

        Parameters
        ----------
        src_path : str or Path object
            Path to HDF5 file for source dataset.
        tgt_path : str or Path object
            Path to HDF5 file for target dataset.
        src_num : int
            Number of samples to use per class for the source dataset.
        tgt_num : int
            Number of samples to use per class for the source dataset.
        sample_ratio : int
            Ratio between the number of intraclass pairs
            (y_src == y_tgt) to interclass pairs (y_src != y_tgt).
        transform : Compose transform containing PyTorch transforms
            Pre-processing operations to apply to images when calling
            __getitem__.
        """
        super().__init__()
        self.transform = transform

        with h5py.File(src_path, "r") as f_s, h5py.File(tgt_path, "r") as f_t:
            # Read datasets from HDF5 file pointers
            src_X, src_y = f_s["X_tr"][()], f_s["y_tr"][()]
            tgt_X, tgt_y = f_t["X_tr"][()], f_t["y_tr"][()]

            # Sample datasets using configuration parameters
            self.X, self.y = {}, {}
            self.X['src'], self.y['src'] = self._resample_data(src_X, src_y,
                                                               src_num)
            self.X['tgt'], self.y['tgt'] = self._resample_data(tgt_X, tgt_y,
                                                               tgt_num)
            self.intra_idxs, self.inter_idxs = self._create_pairs(sample_ratio)
            self.full_idxs = np.concatenate((self.intra_idxs, self.inter_idxs))

            # Sort as to allow shuffling to be performed by the DataLoader
            self.full_idxs = self.full_idxs[np.lexsort((self.full_idxs[:, 1],
                                                        self.full_idxs[:, 0]))]

    def _resample_data(self, X, y, N):
        """Limit sampling to N instances per class."""
        if N > 0:
            # Split labels into set of indexes for each class
            class_idxs = [np.where(y == c)[0] for c in np.unique(y)]

            # Shuffle each of sets of indexes
            [np.random.shuffle(i) for i in class_idxs]

            # Take N indexes, or fewer if total is less than N
            subset_idx = [i[:N] if len(i) >= N else i for i in class_idxs]

            # Use advanced indexing to get subsets of X and y
            idxs = np.array(subset_idx).ravel()
            np.random.shuffle(idxs)
            X, y = X[idxs], y[idxs]

        return X, y

    def _create_pairs(self, sample_ratio):
        """Enforce ratio of inter/intraclass pairs of samples."""
        # Broadcast target/source labels into mesh grid
        # `source` -> (N, 1) broadcast to (N, M)
        # `target` -> (1, M) broadcast to (N, M)
        target, source = np.meshgrid(self.y['tgt'], self.y['src'])

        # Find index pairs (i_S, i_T) for where src_y == tgt_y
        intra_pair_idxs = np.argwhere(source == target)

        # Find index pairs (i_S, i_T) for where src_y != tgt_y
        inter_pair_idxs = np.argwhere(source != target)

        # Randomly sample the interclass pairs to meet desired ratio
        if sample_ratio > 0:
            n_interclass = sample_ratio * len(intra_pair_idxs)
            np.random.shuffle(inter_pair_idxs)
            inter_pair_idxs = inter_pair_idxs[:n_interclass]

            # Sort as to allow shuffling to be performed by the DataLoader
            inter_pair_idxs = inter_pair_idxs[
                np.lexsort((inter_pair_idxs[:, 1], inter_pair_idxs[:, 0]))
            ]

        return intra_pair_idxs, inter_pair_idxs

    def __len__(self):
        """Reflect amount of available pairs of indices."""
        return len(self.full_idxs)

    def __getitem__(self, idx):
        """Get pair of source and target images/labels."""
        src_idx, tgt_idx = self.full_idxs[idx]

        X = {'src': self.X['src'][src_idx], 'tgt': self.X['tgt'][tgt_idx]}
        for key, value in X.items():
            X[key] = self.transform(X[key])

        y = {'src': self.y['src'][src_idx], 'tgt': self.y['tgt'][tgt_idx]}

        return X, y


class SingleDataset(Dataset):
    """Single dataset, used here for evaluating trained network.

    Attributes
    ----------
    transform : Compose transform containing PyTorch transforms
        Pre-processing operations to apply to images when calling
        __getitem__.
    X : PyTorch Tensor (N, H, W, C)
        Images corresponding to samples of source/target datasets.
    y : PyTorch Tensor (N, 1)
        Labels corresponding to samples of source/target datasets.
    """
    def __init__(self, data_path, suffix, transform):
        """Store data and data transforms."""
        super().__init__()
        self.transform = transform

        with h5py.File(data_path, "r") as f_t:
            # Read datasets from HDF5 file pointers
            self.X = f_t[f"X_{suffix}"][()]
            self.y = f_t[f"y_{suffix}"][()]

    def __len__(self):
        """Reflect amount of available examples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get single image/label pair."""
        return self.transform(self.X[idx]), self.y[idx]


class InfLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __getattr__(self, item):
        return getattr(self.data_loader, item)

    def __iter__(self):
        def inf_loop(data_loader):
            for loader in repeat(data_loader):
                yield from loader

        return inf_loop(self.data_loader)


def init_dataloaders(src_path, tgt_path, src_num, tgt_num, sample_ratio,
                     resize_dim, batch_size, shuffle):
    transforms = Compose([
        ToPILImage(),
        Resize(resize_dim),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = PairDataset(src_path, tgt_path, src_num, tgt_num,
                                sample_ratio, transform=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle)

    valid_dataset = SingleDataset(tgt_path, "tr", transform=transforms)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    test_dataset = SingleDataset(tgt_path, "te", transform=transforms)
    test_dataloader = DataLoader(test_dataset, shuffle=shuffle)

    return train_dataloader, valid_dataloader, test_dataloader
