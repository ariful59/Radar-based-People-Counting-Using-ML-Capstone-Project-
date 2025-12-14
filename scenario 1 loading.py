import os
import numpy as np


def load_scenario1_exact(base_path, sample_size=50, signal_length=1280, measurement_size=200):
    """
    Load Scenario 1 IR-UWB radar signals exactly like the paper.

    Args:
        base_path (str): Path to 'scenario 1' folder.
        sample_size (int): Number of signals per sample (default 50).
        signal_length (int): Number of points per signal (default 1280).
        measurement_size (int): Signals per measurement (default 200).

    Returns:
        X (np.ndarray): shape (num_samples, 50, 1280)
        y (np.ndarray): shape (num_samples,)
    """
    X, y = [], []

    for part_folder in sorted(os.listdir(base_path)):
        part_path = os.path.join(base_path, part_folder)
        if not os.path.isdir(part_path):
            continue

        for people_folder in sorted(os.listdir(part_path), key=lambda x: int(x)):
            people_path = os.path.join(part_path, people_folder)
            if not os.path.isdir(people_path):
                continue

            try:
                label = int(people_folder)
            except ValueError:
                continue

            files = sorted(os.listdir(people_path))
            if not files:
                continue

            # Load all frames for this people count
            frames = []
            for f in files:
                file_path = os.path.join(people_path, f)
                try:
                    data = np.loadtxt(file_path)
                    if data.ndim == 1:
                        data = data[np.newaxis, :]
                    # Pad/truncate to signal_length
                    if data.shape[1] < signal_length:
                        padded = np.zeros((data.shape[0], signal_length))
                        padded[:, :data.shape[1]] = data
                        data = padded
                    elif data.shape[1] > signal_length:
                        data = data[:, :signal_length]
                    frames.append(data)
                except Exception as e:
                    print(f"Skipping {file_path}, error: {e}")

            if not frames:
                continue

            measurement = np.vstack(frames)  # shape: (num_signals, 1280)
            num_signals = measurement.shape[0]

            # Only consider full measurements of exactly measurement_size
            for start in range(0, num_signals, measurement_size):
                m = measurement[start:start + measurement_size]
                if m.shape[0] < measurement_size:
                    continue  # skip incomplete measurement

                # Split into 4 samples of 50 signals
                for i in range(0, measurement_size, sample_size):
                    X.append(m[i:i + sample_size])
                    y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


# Usage
BASE = r"C:\Users\SAKSHI\PycharmProjects\CapstoneProject\dataset\scenario 1"
X, y = load_scenario1_exact(BASE)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Example labels:", np.unique(y))

np.save("X_scenario1.npy", X)  # shape: (1760, 50, 1280)
np.save("y_scenario1.npy", y)  # shape: (1760,)