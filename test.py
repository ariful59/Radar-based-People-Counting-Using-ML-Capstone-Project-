
import numpy as np

arr = np.array([
    [1, 3, 5],
    [10, 20, 30]
])

row_means_keep = arr.mean(axis=0, keepdims=True)
row_means_no_keep = arr.mean(axis=0)

print("arr shape:", arr.shape)
print("row_means_keep shape:", row_means_keep.shape)
print("row_means_keep:\n", row_means_keep)
print("row_means_no_keep shape:", row_means_no_keep.shape)
print("row_means_no_keep:", row_means_no_keep)

# DC removal (subtract per-row mean)
dc_removed = arr - row_means_keep
print("dc_removed:\n", dc_removed)



import numpy as np
import matplotlib.pyplot as plt

def bandpass_hamming_1d(arr: np.ndarray, axis: int = 1, cutoff_bins=(5, 750)) -> np.ndarray:
    """Simple FFT-based band-pass filter with Hamming-tapered edges."""
    X = np.fft.rfft(arr, axis=axis)
    n_bins = X.shape[axis]
    low, high = cutoff_bins
    low = max(0, min(low, n_bins - 1))
    high = max(low, min(high, n_bins - 1))

    # Build mask: zeros, Hamming window inside [low, high]
    mask = np.zeros(n_bins, dtype=np.float32)
    mask[low:high+1] = np.hamming(high - low + 1).astype(np.float32)

    # Broadcast mask along chosen axis
    slicer = [None] * arr.ndim
    slicer[axis] = slice(None)
    X_filt = X * mask[tuple(slicer)]

    return np.fft.irfft(X_filt, n=arr.shape[axis], axis=axis)

# ----- demo on a single 1D row (pretend one radar row = 1280 samples) -----
N = 1280
x = np.arange(N)

# low component (slow trend) + mid component (useful) + high noise
low_comp = 0.5 * np.sin(2*np.pi * x / N * 8)        # very low freq
mid_comp = 1.2 * np.sin(2*np.pi * x / N * 80)       # mid freq (keep)
hi_noise = 0.2 * np.sin(2*np.pi * x / N * 300)      # high freq
row = low_comp + mid_comp + hi_noise

# Put it into a 2D shape: (rows=1, columns=N)
arr = row.reshape(1, -1)

# FFT spectrum magnitude for plotting
X = np.fft.rfft(arr, axis=1)
freq_mag = np.abs(X[0])  # magnitude of spectrum for the single row

# Create the same mask used in the function for plotting
n_bins = X.shape[1]
low, high = 5, 200  # choose a narrower mid-band for illustration
mask = np.zeros(n_bins, dtype=np.float32)
mask[low:high+1] = np.hamming(high - low + 1).astype(np.float32)

# Apply filter
arr_filt = bandpass_hamming_1d(arr, axis=1, cutoff_bins=(low, high))
row_filt = arr_filt[0]

# ----- plots -----
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# 1) Original vs filtered signals (time/range domain)
axs[0].plot(row, label="Original (low + mid + high)")
axs[0].plot(row_filt, label="Filtered (band-pass)", alpha=0.9)
axs[0].set_title("Signal: original vs band-pass filtered")
axs[0].set_xlabel("Range bin index")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# 2) FFT magnitude
axs[1].plot(freq_mag, color="tab:blue")
axs[1].set_title("FFT magnitude (spectrum)")
axs[1].set_xlabel("Frequency bin")
axs[1].set_ylabel("|X(f)|")
axs[1].grid(True, alpha=0.3)

# 3) Mask (Hamming-tapered in passband)
axs[2].plot(mask, color="tab:orange")
axs[2].set_title("Band-pass mask (Hamming-tapered edges)")
axs[2].set_xlabel("Frequency bin")
axs[2].set_ylabel("Weight")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
