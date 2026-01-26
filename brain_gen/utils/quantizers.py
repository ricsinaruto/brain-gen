import math
import numpy as np
from typing import Any
import torch
from torch import Tensor
from sklearn.cluster import MiniBatchKMeans

# Precomputed log1p values for common mu values to avoid repeated tensor creation
_LOG1P_CACHE: dict[int, float] = {}


def _get_log1p_mu(mu: int) -> float:
    """Get precomputed log1p(mu) value, caching for repeated calls."""
    if mu not in _LOG1P_CACHE:
        _LOG1P_CACHE[mu] = math.log1p(mu)
    return _LOG1P_CACHE[mu]


def mulaw_torch(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """Torch version of mu-law companding and quantization.

    Optimized version that: - Avoids unnecessary tensor allocations (precomputes
    log1p(mu) as scalar) - Works element-wise without flattening/reshaping - Uses fused
    operations where possible

    Args:     x: Input tensor (float, expected in [-1, 1])     mu: Mu parameter (number
    of quantization levels - 1)

    Returns:     Quantized tensor with values in [0, mu] as long dtype
    """
    # Precomputed scalar division factor (avoids tensor creation per call)
    log1p_mu_inv = 1.0 / _get_log1p_mu(mu)

    # Clip to valid range (in-place on a clone to avoid modifying input)
    x_clipped = x.clamp(-0.999, 0.999)

    # Mu-law compression: sign(x) * log1p(mu * |x|) / log1p(mu)
    # Fused into fewer operations
    abs_x = x_clipped.abs()
    compressed = x_clipped.sign() * torch.log1p(mu * abs_x) * log1p_mu_inv

    # Quantize to integers in [0, mu]
    # (compressed + 1) * 0.5 * mu + 0.5 = compressed * (mu/2) + (mu/2) + 0.5
    half_mu = mu * 0.5
    digitized = (compressed * half_mu + half_mu + 0.5).long()

    return digitized


def mulaw(x: np.ndarray, mu: int = 255) -> tuple[np.ndarray, np.ndarray]:
    # Select smallest uint type that can hold mu value
    if mu <= 255:
        dtype = np.uint8
    elif mu <= 65535:
        dtype = np.uint16
    elif mu <= 4294967295:
        dtype = np.uint32
    else:
        dtype = np.uint64

    # Store original shape but work with flattened array
    shape = x.shape
    x_flat = x.ravel()

    # Compute mu-law compression in one step
    compressed = np.sign(x_flat) * np.log1p(mu * np.abs(x_flat)) / np.log1p(mu)

    # Quantize to integers in [0, mu]
    compressed = (compressed + 1) * 0.5 * mu + 0.5
    digitized = compressed.astype(dtype)

    # perform inverse mu-law compression
    x_recon = mulaw_inv(digitized, mu)

    return digitized.reshape(shape), x_recon.reshape(shape)


def mulaw_inv(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Inverse mu-law companding."""
    # Scale from [0, mu] to [-1, 1]
    x_scaled = (x.astype(np.float32)) / mu * 2 - 1

    # Inverse mu-law transformation
    x_out = np.sign(x_scaled) * (1 / mu) * ((1 + mu) ** np.abs(x_scaled) - 1)

    return x_out


def mulaw_inv_torch(x_int: Tensor, mu: int = 255) -> Tensor:
    """Inverse µ‑law for integer bins ∈ [0, mu].

    Returns float tensor ∈ [‑1, 1].
    """
    mu = float(mu)
    x = x_int.float()
    x_norm = x / mu  # [0,1]
    x_norm = 2 * x_norm - 1  # [‑1,1]
    mag = (1 / mu) * ((1 + mu) ** x_norm.abs() - 1)
    return torch.sign(x_norm) * mag


def adaptive_quantize(
    data: dict[str, Any], n_bits: int = 8, batch_size: int = 1024
) -> dict[str, Any]:
    """Adaptive quantization using k-means clustering for better reconstruction.

    Args:     data: Dictionary containing raw_array     n_bits: Number of bits for
    quantization (default 8 bits = 256 levels)     batch_size: Batch size for mini-batch
    k-means
    """
    n_clusters = 2**n_bits
    raw_shape = data["raw_array"].shape

    # Reshape to 2D array for clustering
    X = data["raw_array"].reshape(-1, 1)

    # Initialize and fit k-means
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, n_init="auto"
    )

    # Fit in batches
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        kmeans.partial_fit(X[i:end_idx])

    # Get cluster centers and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.predict(X)

    # Compute reconstruction
    x_recon = centroids[labels].reshape(raw_shape)

    # Compute quantization error
    data["quantization_error"] = np.mean((data["raw_array"] - x_recon) ** 2)
    print(f"INFO: Quantization MSE: {data['quantization_error']:.6f}")

    # Store quantization parameters
    data["centroids"] = centroids
    data["raw_array"] = labels.reshape(raw_shape)

    return data


def predictive_residual_encode(x: np.ndarray) -> np.ndarray:
    """Second-order predictor residuals with seeds stored in the first two steps.

    ẑ[t] = 2*z[t-1] - z[t-2] e[t] = z[t] - ẑ[t]
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (C, T); got shape {arr.shape}")

    C, T = arr.shape
    residual = np.empty_like(arr, dtype=np.float32)
    if T == 0:
        return residual

    residual[:, 0] = arr[:, 0]
    if T == 1:
        return residual

    residual[:, 1] = arr[:, 1]
    if T > 2:
        residual[:, 2:] = arr[:, 2:] - 0.95 * arr[:, 1:-1]  # + arr[:, :-2]
    return residual


def quantize_deadzone_linear(
    e: np.ndarray, deadzone: float, Emax: float = 1.0, bins: int = 256
) -> np.ndarray:
    q = np.zeros_like(e, dtype=np.uint8)
    center = bins // 2  # pick center code

    # deadzone -> exactly center
    mask = np.abs(e) <= deadzone
    q[mask] = center

    # outside deadzone: quantize linearly to remaining codes
    # map (-Emax..-deadzone) -> (0..127), (deadzone..Emax) -> (129..255)
    pos = e > deadzone
    neg = e < -deadzone

    # positive side
    u = (e[pos] - deadzone) / (Emax - deadzone)  # 0..1
    q[pos] = (center + 1 + np.round(u * (bins - 1 - center - 1))).astype(np.uint8)

    # negative side
    u = (-e[neg] - deadzone) / (Emax - deadzone)  # 0..1
    q[neg] = (center - 1 - np.round(u * (center - 1))).astype(np.uint8)

    return q


def predictive_residual_decode(residual: np.ndarray) -> np.ndarray:
    """Invert the predictor residuals via causal integration."""
    res = np.asarray(residual, dtype=np.float32)
    if res.ndim != 2:
        raise ValueError(f"Expected 2D array (C, T); got shape {res.shape}")

    C, T = res.shape
    out = np.empty_like(res, dtype=np.float32)
    if T == 0:
        return out

    out[:, 0] = res[:, 0]
    if T == 1:
        return out

    out[:, 1] = res[:, 1]
    for t in range(2, T):
        out[:, t] = res[:, t] + 2 * out[:, t - 1] - out[:, t - 2]
    return out


def predictive_mulaw_decode(
    tokens: np.ndarray, *, scale: float | np.ndarray = 1.0, mu: int = 255
) -> np.ndarray:
    """Inverse µ-law then integrate residuals back to continuous values."""
    residual = mulaw_inv(tokens, mu)
    residual = residual * np.asarray(scale, dtype=np.float32)
    return predictive_residual_decode(residual)
