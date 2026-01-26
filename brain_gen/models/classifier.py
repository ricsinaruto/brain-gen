import torch
import torch.nn as nn

from .ntd import NTD
from .bendr import BENDRForecast
from .wavenet import WavenetFullChannel, Wavenet3D
from .tasa3d import TASA3D


def get_model_class(model_name: str):
    """Get model class by name."""
    model_classes = {
        "NTD": NTD,
        "BENDRForecast": BENDRForecast,
        "WavenetFullChannel": WavenetFullChannel,
        "Wavenet3D": Wavenet3D,
        "TASA3D": TASA3D,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_classes[model_name]


class ClassifierContinuous(nn.Module):
    """Generic model that adds an extra classification head to a base forecasting
    model."""

    def __init__(self, base_model_name: str, model_args: dict, num_classes: int):
        super().__init__()
        # Load the base model
        self.base_model = get_model_class(base_model_name)(**model_args)

        # Add a classification head
        self.classifier = nn.Linear(self.base_model.output_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.base_model.encode(x)  # [B, C, T]

        logits = self.classifier(x[..., -1])  # [B, num_classes]
        return logits


class ClassifierQuantized(nn.Module):
    """Generic model that adds an extra classification head to a base forecasting
    model."""

    def __init__(
        self,
        base_model_name: str,
        model_args: dict,
        num_channels: int,
        num_classes: int,
        red_dim: int,
        d_model: int,
    ):
        super().__init__()
        # Load the base model
        self.base_model = get_model_class(base_model_name)(**model_args)

        # Add a classification head
        self.reducer = nn.Linear(d_model, red_dim)
        self.classifier = nn.Linear(num_channels * red_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.base_model(x, return_logits=False)  # B, C, T, d_model
        x = self.reducer(x[:, :, -1, :])  # B, C, red_dim

        # flatten the last dimension
        x = x.reshape(x.shape[0], -1)
        logits = self.classifier(x)  # B, C, num_classes
        return logits


class ClassifierQuantizedImage(ClassifierQuantized):
    """Generic model that adds an extra classification head to a base forecasting
    model."""

    def __init__(
        self,
        num_channels: int,
        red_dim: int,
        image_size: int,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        base_model_name: str,
        model_args: dict,
        num_classes: int,
        quant_levels: int = 256,
    ):
        super().__init__(
            base_model_name,
            model_args,
            num_channels,
            num_classes,
            red_dim,
            quant_levels,
        )
        self.H = image_size
        self.W = image_size

        assert row_idx.shape == col_idx.shape
        self.register_buffer("row_idx", row_idx.long())
        self.register_buffer("col_idx", col_idx.long())

    def forward(self, x: torch.Tensor):
        x = self.base_model(x)  # B, H, W, T, Q

        x = x[:, self.row_idx, self.col_idx, ...]  # B, C, T, Q
        x = self.reducer(x[:, :, -1, :])  # B, C, red_dim

        # flatten the last dimension
        x = x.reshape(x.shape[0], -1)
        logits = self.classifier(x)  # B, C, num_classes
        return logits
