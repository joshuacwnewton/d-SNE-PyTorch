"""Module containing CNN architectures for demonstrating d-SNE."""

# Stdlib imports
from collections import OrderedDict

# Third-party imports
from torch import nn


class LeNetConvBlock(nn.Sequential):
    """Convolution block for LeNetPlus.

    Attributes
    ----------
    image_dim : tuple of ints
        The dimensions (H, W) of the image, updated at each layer.
    in_channels : int
        Number of input channels, updated for each layer.
    out_channels : int
        Number of output channels, unchanging for all layers.

    Methods
    -------
    _update_dimensions :
        Update dimension attributes using layer parameters.
    """

    def __init__(self, n_layers, in_channels, out_channels,
                 conv2d_kernel_size, maxpool2d_kernel_size,
                 conv2d_kwargs={}, leakyrelu_kwargs={}, maxpool2d_kwargs={},
                 image_dim=(0, 0)):
        """Constructor for convolution block used by LeNetPlus.

        Parameters
        ----------
        n_layers : int
            Number of times to repeat Conv2D and LeakyReLU layers.
        in_channels : int
            Number of channels in the input image for the Conv2D layer.
        out_channels : int
            Number of channels produced by the convolution.
        conv2d_kernel_size : int or tuple of ints
            Size of the convolving kernel.
        maxpool2d_kernel_size : int or tuple of ints
            The size of the window to take a max over.
        conv2d_kwargs : dict
            Any remaining keyword arguments for the nn.Conv2d module.
        leakyrelu_kwargs : dict
            Any remaining keyword arguments for the nn.LeakyReLU module.
        maxpool2d_kwargs : dict
            Any remaining keyword arguments for the nn.MaxPool2d module.
        image_dim : tuple of ints
            The dimensions (H, W) of the input image.
        """

        self.image_dim = image_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Alternate Conv2D and LeakyReLU for n_layers, then add final MaxPool2D
        layers = []
        for i in range(n_layers):
            layers.append((f"Conv2d_{i}", nn.Conv2d(self.in_channels,
                                                    self.out_channels,
                                                    conv2d_kernel_size,
                                                    **conv2d_kwargs)))
            self._update_dimemsions(layers[-1][1])
            layers.append((f"LeakyReLU_{i}", nn.LeakyReLU(**leakyrelu_kwargs)))

        layers.append(("MaxPool2d_0", nn.MaxPool2d(maxpool2d_kernel_size,
                                                   **maxpool2d_kwargs)))
        self._update_dimemsions(layers[-1][1])

        super(LeNetConvBlock, self).__init__(OrderedDict(layers))

    def _update_dimemsions(self, layer):
        """Update dimension attributes using layer parameters."""
        P, K, S = layer.padding, layer.kernel_size, layer.stride
        (P_h, P_w) = P if type(P) is tuple else (P, P)
        (K_h, K_w) = K if type(K) is tuple else (K, K)
        (S_h, S_w) = S if type(S) is tuple else (S, S)

        self.in_channels = getattr(layer, "out_channels", self.out_channels)
        self.image_dim = (int(((self.image_dim[0] - K_h + 2*P_h) / S_h) + 1),
                          int(((self.image_dim[1] - K_w + 2*P_w) / S_w) + 1))


class LeNetPlus(nn.Module):
    """Default network architecture used by d-SNE.

    Attributes
    ----------
    model : nn.Sequential
        Feature extraction and regularization layers in network.
    output_0_features : nn.Linear
        Fully-connected layer which outputs feature vectors.
    output_1_classes : nn.Linear
        Fully-connected layer which outputs class scores.

    Methods
    -------
    forward :
        Apply layers in sequence to input mini-batch.

    Notes
    -----
    The d-SNE publication explicitly mentions using VGG-16 and
    ResNet-101 to compare with other SOTA methods, FADA and CCSA.
    However, the provided code in d-SNE's public repository defaults to
    this architecture.

    https://github.com/ShownX/d-SNE

    I cannot find reference to "LeNetPlus" anywhere but this repository.
    I have reached out to the author for further clarification on the
    configuration for which the published results were recorded.

    https://github.com/aws-samples/d-SNE/issues/13
    """

    def __init__(self, input_dim, classes=10, feature_size=256,
                 dropout=0.5, use_bn=False, use_inn=False):
        """Constructor for the LeNetPlus model architecture.

        Parameters
        ----------
        input_dim : int or tuple of ints (C, H, W)
            Expected dimensions for input images.
        classes : int
            Number of classes which defines the final output dimensions.
        feature_size : int
            Length of the feature vectors extracted by the network.
        dropout : float
            Hyperparameter to use for dropout layer. Setting to <= 0
            disables this layer.
        use_bn : bool
            Flag for whether batch normalization should be used at
            the start of the inner level of the network.
        use_inn : bool
            Flag for whether instance normalization should be used at
            the start of the outer level of the network.
        """
        super(LeNetPlus, self).__init__()
        if isinstance(input_dim, int):
            (D_in, cur_h, cur_w) = (3, input_dim, input_dim)
        else:
            (D_in, cur_h, cur_w) = input_dim

        layers = []
        if use_inn:
            layers.append(("InstanceNorm2d_0",
                           nn.InstanceNorm2d(num_features=D_in)))

        for i, D_out in enumerate([32, 64, 128]):
            if use_bn:
                layers.append((f"BatchNorm2d_{i}",
                               nn.BatchNorm2d(num_features=D_in)))

            layers.append((f"ConvBlock_{i}", LeNetConvBlock(
                n_layers=2,
                in_channels=D_in,
                out_channels=D_out,
                conv2d_kernel_size=3,
                maxpool2d_kernel_size=2,
                conv2d_kwargs={"padding": 2},
                leakyrelu_kwargs={"negative_slope": 0.2},
                maxpool2d_kwargs={"stride": 2},
                image_dim=(cur_h, cur_w)
            )))

            # Update post-convolution dimensions
            D_in = layers[-1][1].in_channels
            cur_h, cur_w = layers[-1][1].image_dim

            if dropout > 0:
                layers.append((f"Dropout_{i}", nn.Dropout2d(dropout)))

        self.model = nn.Sequential(OrderedDict(layers))
        self.output_0_features = nn.Linear(D_out*cur_h*cur_w, feature_size)
        self.output_1_classes = nn.Linear(feature_size, classes)

    def forward(self, x):
        """Compute the forward pass for the network."""
        # Convolution block layers
        x = self.model(x)

        # Fully connected layers
        features = self.output_0_features(x.reshape(len(x), 1, -1).squeeze())
        outputs = self.output_1_classes(features)

        return features, outputs
