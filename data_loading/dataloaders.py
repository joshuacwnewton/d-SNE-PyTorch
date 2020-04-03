from torch.utils.data import DataLoader

from data_loading.datasets import PairDataset, SingleDataset


def get_dsne_dataloaders(src_path, src_X_name, src_y_name,
                         tgt_path, tgt_X_name, tgt_y_name):

    # TODO: Config options found in d-SNE code:
    #   -DigitDataset
    #       -TARGET_PATH
    #   -DigitPairsDataset
    #       -SOURCE_PATH
    #       -SOURCE_NUM
    #       -TARGET_PATH
    #       -TARGET_NUM
    #       -SAMPLE_RATIO

    # Pass loaded datasets into Dataset objects
    train_dataset = PairDataset(src_path, src_X_name, src_y_name,
                                tgt_path, tgt_X_name, tgt_y_name)

    # TODO: Implement SingleDataset
    # test_dataset = SingleDataset(data=mnist_m_path)

    # Pass Dataset objects to DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    # test_dataloader = DataLoader(test_dataset)

    return train_dataloader  #, test_dataloader
