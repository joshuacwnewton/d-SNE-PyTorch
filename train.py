from data_loading.dataloaders import get_dsne_dataloaders
from model.networks import LeNetPlus


def main():
    train_dataloader = get_dsne_dataloaders("data_loading/data/mnist.h5",
                                            "X_tr", "y_tr",
                                            "data_loading/data/mnist_m.h5",
                                            "X_tr", "y_tr")
    model = LeNetPlus()


if __name__ == "__main__":
    # TODO: Argparsing (dataset selection, hyperparameters, etc.)

    main()