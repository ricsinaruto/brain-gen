from typing import Optional

import torch
from torch import nn


class CNNMultivariate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 5,
        linear: bool = False,
        groups: int = 1,
    ):
        """Args:

        in_channels: Number of input channels num_layers: Number of layers
        hidden_channels: Number of hidden channels univariate: Whether to handle
        channels as the batch dimension kernel_size: Kernel size linear: Whether to use
        a linear layer instead of a convolutional layer groups: Number of groups for the
        convolutional layer. Set this to the number of input channels to have a per-
        channel model.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        # For causal padding, we use padding modules for left padding only
        padding_size = kernel_size - 1

        final_conv = None
        if in_channels != hidden_channels:
            final_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.ConstantPad1d((padding_size, 0), 0))
            layers.append(
                nn.Conv1d(in_channels, hidden_channels, kernel_size, groups=groups)
            )
            in_channels = hidden_channels

            if not linear:
                layers.append(nn.ReLU())

        if final_conv is not None:
            layers.append(final_conv)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class CNNUnivariate(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        hidden_channels: int = 1,
        kernel_size: int = 5,
        linear: bool = False,
    ):
        """Args:

        num_layers: Number of layers hidden_channels: Number of hidden channels
        kernel_size: Kernel size linear: Whether to use a linear layer instead of a
        convolutional layer
        """
        super().__init__()

        self.cnn = CNNMultivariate(1, num_layers, hidden_channels, kernel_size, linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # put channels as batch dimension
        shape = x.shape
        x = x.reshape(-1, 1, shape[-1])

        x = self.cnn(x)

        return x.reshape(*shape)


class CNNMultivariateQuantized(nn.Module):
    """Simple multivariate CNN model applied to quantized data."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_embeddings: int = 1,
        num_layers: int = 1,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 5,
        linear: bool = False,
        groups: int = 1,
    ):
        """Args:

        in_channels: Number of input channels num_classes: Number of classes
        num_embeddings: Number of embeddings num_layers: Number of layers
        hidden_channels: Number of hidden channels kernel_size: Kernel size linear:
        Whether to use a linear layer instead of a convolutional layer groups: Number of
        groups for the convolutional layer. Set this to the number of input channels to
        have a per-channel model.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_classes, num_embeddings)

        self.cnn = CNNMultivariate(
            in_channels * num_embeddings,
            num_layers,
            hidden_channels,
            kernel_size,
            linear,
            groups=groups,
        )

        self.head = nn.Linear(num_embeddings, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.embedding(x).permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = self.cnn(x)

        x = x.reshape(x.shape[0], -1, self.num_embeddings, x.shape[-1])
        x = x.permute(0, 1, 3, 2)

        return self.head(x)


class CNNUnivariateQuantized(CNNMultivariateQuantized):
    def __init__(
        self,
        num_classes: int,
        num_embeddings: int = 1,
        num_layers: int = 1,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 5,
        linear: bool = False,
    ):
        """Args:

        num_classes: Number of classes num_embeddings: Number of embeddings num_layers:
        Number of layers hidden_channels: Number of hidden channels kernel_size: Kernel
        size linear: Whether to use a linear layer instead of a convolutional layer
        """
        super().__init__(
            1,
            num_classes,
            num_embeddings,
            num_layers,
            hidden_channels,
            kernel_size,
            linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.embedding(x).permute(0, 1, 3, 2)
        shape = x.shape
        x = x.reshape(-1, self.num_embeddings, x.shape[-1])

        x = self.cnn(x)

        x = x.reshape(shape)
        x = x.permute(0, 1, 3, 2)

        return self.head(x)
