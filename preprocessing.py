
import numpy as np
from scipy.signal import lfilter

def dc_remove(arr: np.ndarray) -> np.ndarray:
    """Subtract per-row mean (DC removal)."""
    return arr - arr.mean(axis=1, keepdims=True)

def bandpass_hamming_1d(arr: np.ndarray, axis: int = 1, cutoff_bins=(5, 750), use_hamming: bool = False) -> np.ndarray:
    """
    FFT-based band-pass filter.
    If use_hamming is True, applies a Hamming window over the selected bins.
    Otherwise, uses a rectangular mask (standard ideal bandpass).
    """ 
    X = np.fft.rfft(arr, axis=axis)
    n_bins = X.shape[axis]
    low, high = cutoff_bins
    low = max(0, min(low, n_bins - 1))
    high = max(low, min(high, n_bins - 1))
    
    mask = np.zeros(n_bins)
    if use_hamming:
        mask[low:high+1] = np.hamming(high - low + 1)
    else:
        mask[low:high+1] = 1.0
        
    slicer = [None] * arr.ndim
    slicer[axis] = slice(None)
    X *= mask[tuple(slicer)]
    return np.fft.irfft(X, n=arr.shape[axis], axis=axis)

def clutter_remove(arr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    if arr.shape[0] == 0:
        return arr
    
    b = [1 - alpha]
    a = [1, -alpha]
    
    zi = alpha * arr[0:1, :]
    C, _ = lfilter(b, a, arr, axis=0, zi=zi)
    
    return arr - C
