"""Dataloaders for sampling and batching datasets."""

# Stdlib imports
from itertools import repeat

# Third-party imports
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, ToPILImage, ToTensor,
                                    Resize, Normalize)

# Local application imports
from dsne_pytorch.data_loading.datasets import PairDataset, SingleDataset


def get_dsne_dataloaders(src_path, tgt_path, src_num, tgt_num, sample_ratio,
                         image_dim, batch_size, shuffle):
    transforms = Compose([
        ToPILImage(),
        Resize(image_dim),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = PairDataset(src_path, tgt_path, src_num, tgt_num,
                                sample_ratio, transform=transforms)
    test_dataset = SingleDataset(tgt_path, transform=transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, shuffle=shuffle)

    return train_dataloader, test_dataloader


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
