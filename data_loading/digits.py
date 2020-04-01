from torch.utils.data import DataLoader

from data_loading.datasets import PairDataset, SingleDataset


def mnist_to_mnistm(mnist_path="./data/MNIST.bin",
                    mnistm_path="./data/MNIST-M.bin"):

    # TODO: Config options found in d-SNE code:
    #   -DigitDataset
    #       -TARGET_PATH
    #   -DigitPairsDataset
    #       -SOURCE_PATH
    #       -SOURCE_NUM
    #       -TARGET_PATH
    #       -TARGET_NUM
    #       -SAMPLE_RATIO

    # load serialized mnist images / labels
    mnist = mnist_path
    # Sample form: {
    #     "X_tr": None,
    #     "y_tr": None,
    #     "X_te": None,
    #     "y_te": None
    # }

    # load serialized mnist-m images / labels
    mnistm = mnistm_path
    # Sample form: {
    #     "X_tr": None,
    #     "y_tr": None,
    #     "X_te": None,
    #     "y_te": None
    # }

    # pass loaded datasets into Dataset objects
    train_dataset = PairDataset(data_src=mnist, data_tgt=mnistm)
    test_dataset = SingleDataset(data=mnistm)

    # pass Dataset objects to DataLoader objects
    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)

    # return training and testing dataloaders
    return train_dataloader, test_dataloader