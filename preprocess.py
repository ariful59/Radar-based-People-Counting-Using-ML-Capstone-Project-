import numpy as np
from scipy.signal import butter, filtfilt

X = np.load(r"/Radar-based-People-Counting-Using-ML-Capstone-Project-/dataset\X_scenario1.npy")
y = np.load(r"/Radar-based-People-Counting-Using-ML-Capstone-Project-/dataset\y_scenario1.npy")

def bandpass_filter(signal, fs=20e9, lowcut=1e9, highcut=9.9e9, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.9999  # ensure it’s < 1
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def refine_signal(signal, start_idx=50, end_idx=1280):
    """
    Remove near-range static reflections (antenna coupling / ring-down)
    Args:
        signal: 1D array of shape (1280,)
        start_idx: index to start keeping (skip near-range)
        end_idx: index to stop (optional)
    Returns:
        refined_signal: 1D array of length (end_idx - start_idx)
    """
    refined = signal[start_idx:end_idx]
    return refined

def preprocess_X(X, start_idx=50, end_idx=1280, fs=20e9, lowcut=1e9, highcut=10e9):
    """
    Preprocess all radar samples
    Args:
        X: (num_samples, 50, 1280) raw radar data
    Returns:
        X_preprocessed: (num_samples, 50, L) preprocessed radar matrices
    """
    num_samples, num_frames, num_points = X.shape
    L = end_idx - start_idx
    X_preprocessed = np.zeros((num_samples, num_frames, L))

    for i in range(num_samples):
        for j in range(num_frames):
            signal = X[i, j, :]
            filtered = bandpass_filter(signal, fs=fs, lowcut=lowcut, highcut=highcut)
            refined = refine_signal(filtered, start_idx=start_idx, end_idx=end_idx)
            X_preprocessed[i, j, :] = refined

    return X_preprocessed

# Usage example
X_preprocessed = preprocess_X(X, start_idx=50, end_idx=1280)
print("Preprocessed X shape:", X_preprocessed.shape)

np.save("X_preprocessed.npy", X_preprocessed)
np.save("y.npy", y)