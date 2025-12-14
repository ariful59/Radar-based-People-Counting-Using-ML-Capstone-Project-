import numpy as np
import pywt  # For wavelet transform
from sklearn.preprocessing import StandardScaler

X_preprocessed = np.load(r"/preprocessing/X_preprocessed.npy")
y = np.load(r"/preprocessing/y.npy")

def extract_ctf_features(sample, wavelet='db1', levels=2):
    """
    Extract curvelet-like features from a 2D radar sample (50x1230)
    Args:
        sample: 2D array (num_frames, num_points)
    Returns:
        feature_vector: 1D array
    """
    coeffs = pywt.wavedec2(sample, wavelet=wavelet, level=levels)
    features = []
    for level in coeffs:
        if isinstance(level, tuple):
            # level contains (cH, cV, cD)
            for arr in level:
                features.append(np.mean(arr))
                features.append(np.std(arr))
        else:
            # Approximation coefficients
            features.append(np.mean(level))
            features.append(np.std(level))
    return np.array(features)

def extract_dbf_features(sample, num_bins=30):
    """
    Distance Bin Features
    Args:
        sample: 2D array (num_frames, num_points)
        num_bins: how many bins along distance axis
    Returns:
        feature_vector: 1D array
    """
    num_points = sample.shape[1]
    bin_size = num_points // num_bins
    features = []
    for i in range(num_bins):
        bin_slice = sample[:, i*bin_size:(i+1)*bin_size]
        features.append(np.mean(bin_slice))
        features.append(np.std(bin_slice))
    return np.array(features)

def extract_hybrid_features(X):
    """
    Extract hybrid CTF + DBF features for all samples
    Args:
        X: (num_samples, num_frames, num_points)
    Returns:
        X_features: (num_samples, feature_dim)
    """
    all_features = []
    for sample in X:
        ctf = extract_ctf_features(sample)
        dbf = extract_dbf_features(sample)
        hybrid = np.concatenate([ctf, dbf])
        all_features.append(hybrid)
    X_features = np.array(all_features)
    # Optional: standardize features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    return X_features

# Usage
X_features = extract_hybrid_features(X_preprocessed)
print("Hybrid feature matrix shape:", X_features.shape)

np.save("X_features.npy", X_features)
np.save("y.npy", y)
