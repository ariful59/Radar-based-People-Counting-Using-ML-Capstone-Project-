
import numpy as np

def dc_remove(arr: np.ndarray) -> np.ndarray:
    """Subtract per-row mean (DC removal)."""
    return arr - arr.mean(axis=1, keepdims=True)

def bandpass_hamming_1d(arr: np.ndarray, axis: int = 1, cutoff_bins=(5, 750)) -> np.ndarray:
    """Simple FFT-based band-pass filter."""
    X = np.fft.rfft(arr, axis=axis)
    n_bins = X.shape[axis]
    low, high = cutoff_bins
    low = max(0, min(low, n_bins - 1))
    high = max(low, min(high, n_bins - 1))
    mask = np.zeros(n_bins)
    mask[low:high+1] = np.hamming(high - low + 1)
    slicer = [None] * arr.ndim
    slicer[axis] = slice(None)
    X *= mask[tuple(slicer)]
    return np.fft.irfft(X, n=arr.shape[axis], axis=axis)

def clutter_remove(arr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Exponential clutter removal across rows."""
    s = np.empty_like(arr)
    C = np.zeros_like(arr[0:1, :])
    for t in range(arr.shape[0]):
        x_t = arr[t:t+1, :]
        C = alpha * C + (1 - alpha) * x_t
        s[t:t+1, :] = x_t - C
    return s
