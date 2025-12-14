"""
Preprocessing module for IR-UWB Radar People Counting Dataset
"""

import numpy as np
from scipy.signal import savgol_filter


def preprocess_signal(signal):
    """
    Preprocess a single IR-UWB radar signal.

    Parameters
    ----------
    signal : np.ndarray
        Radar signal of shape (200, 1280)

    Returns
    -------
    np.ndarray
        Preprocessed radar signal
    """

    # Step 1: Static clutter removal
    signal = signal - np.mean(signal, axis=0)

    # Step 2: Normalization
    signal = signal / (np.std(signal) + 1e-8)

    # Step 3: Noise smoothing
    signal = savgol_filter(signal, window_length=11, polyorder=2, axis=0)

    return signal
