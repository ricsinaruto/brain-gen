import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def plot_psd(data: np.ndarray, sfreq: float, prefix: str) -> plt.Figure:
    """Plot the PSD of the data.

    Args:     data: Array of shape (C, T) containing the data     sfreq: Sampling
    frequency     prefix: Prefix for the filenames

    Returns:     fig: Figure object
    """
    # Use Welch's method for a smoother PSD estimate
    # Choose segment length up to 1024 samples or full length if shorter
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        axis=-1,
        nperseg=sfreq,
        scaling="density",
    )  # psd shape: (C, len(freqs))

    # PSD plot with automatic y-axis limits to avoid outlier distortion
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(freqs, psd.T, alpha=0.3)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Power")
    ax.set_title(f"PSD - {prefix}")
    ax.set_yscale("log")

    # Set y-limits based on the 1st and 99th percentiles to avoid outlier distortion
    psd_flat = psd.flatten()
    psd_flat = psd_flat[psd_flat > 0]  # avoid log(0) issues
    if psd_flat.size > 0:
        lower = np.percentile(psd_flat, 0.1)
        upper = np.percentile(psd_flat, 99.9)
        if lower > 0 and upper > lower:
            ax.set_ylim([lower, upper])

    return fig
