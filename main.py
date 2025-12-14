import sys
from pathlib import Path
import io
import numpy as np
import rarfile
from preprocessing import dc_remove, bandpass_hamming_1d, clutter_remove


def parse_text_to_array(text: str) -> np.ndarray:
    """Convert text (CSV or whitespace-separated numbers) to NumPy array."""
    bio = io.StringIO(text)
    try:
        arr = np.loadtxt(bio, delimiter=",")
    except Exception:
        bio.seek(0)
        arr = np.loadtxt(bio)  # whitespace fallback
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def main():
    files_dir = Path.cwd() / "files"
    if not files_dir.is_dir():
        print(f"Directory not found: {files_dir}")
        sys.exit(1)

    rar_files = list(files_dir.glob("*.rar"))
    if not rar_files:
        print("No .rar files found.")
        sys.exit(0)

    all_processed = []

    for rar_path in rar_files:
        print(f"\nProcessing: {rar_path.name}")
        with rarfile.RarFile(rar_path) as rf:
            for info in rf.infolist():
                if info.is_dir():
                    continue

                data = rf.read(info)
                try:
                    text = data.decode('utf-8')  # decode as text
                except UnicodeDecodeError:
                    print(f"Skipping {info.filename} (not text)")
                    continue

                # Convert text to numeric array
                try:
                    arr = parse_text_to_array(text)
                except Exception as e:
                    print(f"Skipping {info.filename}: cannot parse numeric data ({e})")
                    continue

                # === Preprocessing ===
                arr = dc_remove(arr)
                arr = bandpass_hamming_1d(arr, axis=1, cutoff_bins=(5, 750))
                arr = clutter_remove(arr, alpha=0.6)

                all_processed.append(arr)
                print(f"Processed {info.filename} -> shape {arr.shape}")

    print(f"\nTotal processed samples: {len(all_processed)}")

    # Later: feed `all_processed` into CNN training


if __name__ == "__main__":
    main()
