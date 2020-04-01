from torch.utils.data import DataLoader

from data_loading.datasets import PairDataset, SingleDataset


def mnist_to_mnistm(mnist_path="./data/mnist.h5",
                    mnist_m_path="./data/mnist_m.h5"):

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
    train_dataset = PairDataset(mnist_path, "X_tr", "y_tr",
                                mnist_m_path, "X_tr", "y_tr")

    # TODO: Implement SingleDataset
    # test_dataset = SingleDataset(data=mnist_m_path)

    # Pass Dataset objects to DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    # test_dataloader = DataLoader(test_dataset)

    return train_dataloader  #, test_dataloader


if __name__ == "__main__":
    # Temporary call to test PairDataset creation
    train_dataloader = mnist_to_mnistm()
