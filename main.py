
import sys
from pathlib import Path
import io
import numpy as np
import rarfile

from preprocessing import dc_remove, bandpass_hamming_1d, clutter_remove


def parse_text_to_array(text: str) -> np.ndarray:
    """
    Convert text (CSV or whitespace-separated numbers) to a 2D NumPy array.
    - Tries comma-separated first, then whitespace.
    """
    bio = io.StringIO(text)
    try:
        arr = np.loadtxt(bio, delimiter=",")
    except Exception:
        bio.seek(0)
        arr = np.loadtxt(bio)  # whitespace fallback
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32)

def extract_label_from_path(entry_name: str) -> int:
    """
    Given an archive entry path like '3/sample_0001.txt' or '7/abc/xyz.csv',
    return the first folder name that is purely digits as the label.
    """
    # Normalize separators to '/' and split
    parts = entry_name.replace("\\", "/").split("/")
    # Scan folders for a purely-numeric component
    for p in parts:
        if p.isdigit():
            return int(p)
    raise ValueError(f"Could not infer label from path: {entry_name}")

# ---------- main ----------

def main():
    files_dir = Path.cwd() / "files"
    if not files_dir.is_dir():
        print(f"Directory not found: {files_dir}")
        sys.exit(1)

    rar_files = list(files_dir.glob("*.rar"))
    if not rar_files:
        print("No .rar files found.")
        sys.exit(0)

    all_processed = []  # List[np.ndarray], each (200, 1280) (or 2D matrix)
    all_labels = []     # List[int], derived from folder names (0..10)

    for rar_path in rar_files:
        print(f"\n=== Processing RAR: {rar_path.name} ===")
        with rarfile.RarFile(rar_path) as rf:
            for info in rf.infolist():
                # Skip directories
                if info.is_dir():
                    continue

                # Infer label from the entry path (e.g., '7/xxx.txt' -> label 7)
                try:
                    label = extract_label_from_path(info.filename)
                except Exception as e:
                    # If the entry doesn't live under a numeric folder, skip it
                    continue

                # Read entry bytes
                data = rf.read(info)

                # Decode as text (CSV/whitespace numbers). If not text, skip.
                try:
                    text = data.decode('utf-8')
                except UnicodeDecodeError:
                    continue

                # Parse to numeric 2D array
                try:
                    arr = parse_text_to_array(text)
                except Exception as e:
                    print(f"[skip] {info.filename}: cannot parse numeric data ({e})")
                    continue

                # ---------- preprocessing pipeline ----------
                arr = dc_remove(arr)
                arr = bandpass_hamming_1d(arr, axis=1, cutoff_bins=(5, 750))
                arr = clutter_remove(arr, alpha=0.6)

                # save data
                all_processed.append(arr)
                all_labels.append(label)

                print(f"Processed {info.filename} -> label {label}, shape {arr.shape}")

    print(f"\nTotal processed samples: {len(all_processed)} | Total labels: {len(all_labels)}")
    if len(all_processed) != len(all_labels):
        print("[warning] Mismatch in samples vs labels lengths.")

    # >>> Later: feed `all_processed` and `all_labels` into your CNN training <<<
    # e.g., from train_cnn_200x1280 import run_training
    # num_classes = len(set(all_labels))
    # run_training(all_processed, all_labels, num_classes)

if __name__ == "__main__":
    main()
