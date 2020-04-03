from collections import OrderedDict

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