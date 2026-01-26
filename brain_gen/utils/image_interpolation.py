import numpy as np
import torch


def compute_layout_indices(
    pos_2d: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map sensor coordinates to integer pixel indices on a square grid.

    Args:     pos_2d: Array of shape (C, 2) containing sensor x/y coordinates.
    image_size: Target image size (height = width).

    Returns:     row_idx, col_idx: Integer indices in [0, image_size).
    """
    pos = pos_2d.astype(np.float32)
    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    span = pos_max - pos_min
    span[span == 0] = 1.0
    pos_norm = (pos - pos_min) / span

    col_idx = np.round(pos_norm[:, 0] * (image_size - 1)).astype(np.int64)
    row_idx = np.round(pos_norm[:, 1] * (image_size - 1)).astype(np.int64)
    return row_idx, col_idx


def _median_nearest_neighbor(layout_px: np.ndarray) -> float:
    """Compute the median nearest-neighbor distance on the pixel grid."""
    layout_px = layout_px.astype(np.float64, copy=False)
    if len(layout_px) < 2:
        return 1.0

    diff = layout_px[:, None, :] - layout_px[None, :, :]
    d2 = (diff**2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.min(axis=1)
    median = float(np.sqrt(np.median(nn[np.isfinite(nn)])))
    return max(median, 1.0)


class GaussianSensorInterpolator:
    """Smoothly interpolate sparse sensor values onto a dense image grid using Gaussian
    kernels centred at each sensor location.

    The kernel width is derived from the median nearest-neighbour distance to give a
    data-driven smoothing scale.
    """

    def __init__(
        self,
        pos_2d: np.ndarray,
        image_size: int = 32,
        sigma_scale: float = 0.75,
        r_max_factor: float = 4.0,
    ) -> None:
        self.image_size = int(image_size)
        self.row_idx, self.col_idx = compute_layout_indices(pos_2d, self.image_size)
        layout_px = np.stack([self.row_idx, self.col_idx], axis=1)

        sigma = sigma_scale * _median_nearest_neighbor(layout_px)
        self.sigma = float(sigma)
        self.r_max = float(r_max_factor * sigma)

        # Pre-compute Gaussian weights from each sensor to each pixel.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.image_size), torch.arange(self.image_size), indexing="ij"
        )
        layout = torch.tensor(layout_px, dtype=torch.float32)  # (C, 2)

        dy = grid_y[None, :, :] - layout[:, 0][:, None, None]
        dx = grid_x[None, :, :] - layout[:, 1][:, None, None]
        d2 = dx.pow(2) + dy.pow(2)  # (C, H, W)

        weights = torch.exp(-0.5 * d2 / (self.sigma**2))
        if self.r_max > 0:
            weights = torch.where(
                d2 <= (self.r_max**2), weights, torch.zeros_like(weights)
            )

        weight_sum = weights.sum(dim=0)
        self.weights = weights
        self.weight_sum = weight_sum.clamp_min(1e-6)

        self.weights_inv = weights.clone()
        self.sensor_weight_sum = weights.sum(dim=(1, 2)).clamp_min(1e-6)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Args: values: Tensor of shape (C, T) with sensor values.

        Returns:     img: Tensor of shape (T, H, W) representing interpolated frames.
        """
        if values.ndim != 2:
            raise ValueError(
                f"Expected sensor values with shape (C, T), got {values.shape}"
            )

        if self.weights.device != values.device or self.weights.dtype != values.dtype:
            self.weights = self.weights.to(values.device, values.dtype)
            self.weight_sum = self.weight_sum.to(values.device, values.dtype)

        img = torch.einsum("ct,chw->thw", values, self.weights)
        img = img / self.weight_sum
        return img

    def inverse(self, img: torch.Tensor) -> torch.Tensor:
        """Extract sensor values from interpolated images (rough inverse).

        For each sensor, computes a weighted average of image pixels using the same
        Gaussian weights as the forward pass. This is not an exact inverse but provides
        a reasonable approximation.

        Args:     img: Tensor of shape (T, H, W) representing interpolated frames.

        Returns:     values: Tensor of shape (B, C, T) with estimated sensor values.
        """
        img = img.squeeze()
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            raise ValueError(f"Expected image with shape (B, T, H, W), got {img.shape}")

        if self.weights_inv.device != img.device or self.weights_inv.dtype != img.dtype:
            self.weights_inv = self.weights_inv.to(img.device, img.dtype)  # (C, H, W)
            self.sensor_weight_sum = self.sensor_weight_sum.to(img.device, img.dtype)

        # Weighted average of image pixels for each sensor
        values = torch.einsum("bthw,chw->bct", img, self.weights_inv)
        values = values / self.sensor_weight_sum[None, :, None]
        return values


class GaussianSensorInverseInterpolator:
    """Standalone inverse interpolator that extracts sensor values from images.

    This class provides the same inverse functionality as
    GaussianSensorInterpolator.inverse() but can be instantiated independently when only
    the inverse operation is needed.
    """

    def __init__(
        self,
        pos_2d: np.ndarray,
        image_size: int = 32,
        sigma_scale: float = 0.75,
        r_max_factor: float = 4.0,
    ) -> None:
        self.image_size = int(image_size)
        self.n_channels = len(pos_2d)
        self.row_idx, self.col_idx = compute_layout_indices(pos_2d, self.image_size)
        layout_px = np.stack([self.row_idx, self.col_idx], axis=1)

        sigma = sigma_scale * _median_nearest_neighbor(layout_px)
        self.sigma = float(sigma)
        self.r_max = float(r_max_factor * sigma)

        # Pre-compute Gaussian weights (same as forward interpolator)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.image_size), torch.arange(self.image_size), indexing="ij"
        )
        layout = torch.tensor(layout_px, dtype=torch.float32)

        dy = grid_y[None, :, :] - layout[:, 0][:, None, None]
        dx = grid_x[None, :, :] - layout[:, 1][:, None, None]
        d2 = dx.pow(2) + dy.pow(2)

        weights = torch.exp(-0.5 * d2 / (self.sigma**2))
        if self.r_max > 0:
            weights = torch.where(
                d2 <= (self.r_max**2), weights, torch.zeros_like(weights)
            )

        self.weights = weights  # (C, H, W)
        self.sensor_weight_sum = weights.sum(dim=(1, 2)).clamp_min(1e-6)  # (C,)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Extract sensor values from interpolated images.

        Args:     img: Tensor of shape (T, H, W) representing interpolated frames.

        Returns:     values: Tensor of shape (C, T) with estimated sensor values.
        """
        if img.ndim != 3:
            raise ValueError(f"Expected image with shape (T, H, W), got {img.shape}")

        weights = self.weights.to(img.device, img.dtype)
        sensor_weight_sum = self.sensor_weight_sum.to(img.device, img.dtype)

        values = torch.einsum("thw,chw->ct", img, weights)
        values = values / sensor_weight_sum[:, None]
        return values
