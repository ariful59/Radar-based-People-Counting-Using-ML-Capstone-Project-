
import sys
from pathlib import Path
import io
import numpy as np
import rarfile
from exceptiongroup import catch

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

def load_full_data():
    files_dir = Path.cwd() / "files"
    if not files_dir.is_dir():
        print(f"Directory not found: {files_dir}")
        return [], []

    rar_files = list(files_dir.glob("*.rar"))
    if not rar_files:
        print("No .rar files found.")
        return [], []

    all_processed = []  
    all_labels = []     

    for rar_path in rar_files:
        print(f"\n=== Processing RAR: {rar_path.name} ===")
        with rarfile.RarFile(rar_path) as rf:
            for info in rf.infolist():
                if info.is_dir():
                    continue

                try:
                    label = extract_label_from_path(info.filename)
                except Exception:
                    continue

                data = rf.read(info)
                try:
                    text = data.decode('utf-8')
                except UnicodeDecodeError:
                    continue

                try:
                    arr = parse_text_to_array(text)
                except Exception as e:
                    print(f"[skip] {info.filename}: cannot parse numeric data ({e})")
                    continue

                # ---------- preprocessing pipeline ----------
                arr = dc_remove(arr)
                arr = bandpass_hamming_1d(arr, axis=1, cutoff_bins=(5, 750))
                arr = clutter_remove(arr, alpha=0.6)

                # ---------- 2.5s windowing (50 rows) ----------
                window_size = 50
                for i in range(0, 200, window_size):
                    window = arr[i:i+window_size, :]
                    if window.shape[0] == window_size:
                        all_processed.append(window)
                        all_labels.append(label)

                print(f"Processed {info.filename} -> label {label}, split into {200//window_size} windows")
    return all_processed, all_labels

def main():
    data_cache = Path("data_cache.npz")
    
    if not data_cache.exists():
        print("Data cache not found. Processing raw files...")
        all_processed, all_labels = load_full_data()
        if not all_processed:
            print("No data loaded. Exiting.")
            return
        # Save cache for next time
        np.savez(data_cache, data=np.array(all_processed), labels=np.array(all_labels))
    else:
        print(f"Loading cached data from {data_cache}...")
        with np.load(data_cache) as data:
            all_processed = list(data['data'])
            all_labels = list(data['labels'])

    print(f"\nTotal samples: {len(all_processed)} | Total labels: {len(all_labels)}")

    if len(all_processed) != len(all_labels):
        print("[warning] Mismatch in samples vs labels lengths.")

    # Choose your model here:
    # Option A: Custom CNN
    from cnn_model import run_training
    # Option B: Pre-trained ResNet-18
    # from pretrained_model import run_pretrained_training as run_training
    # Option C: Pre-trained Inception-v3 (uncomment to use)
    # from inception_model import run_inception_training as run_training

    print(f"Classes: {sorted(set(all_labels))}")

    # For Custom CNN (Option A):
    from cnn_model import run_training
    # from pretrained_model import run_pretrained_training
    from inception_model import run_inception_training

    ckpt_path = run_inception_training(
        X=all_processed,
        y=all_labels,
        batch_size=16,
        lr=1e-3,
        epochs=10,
        val_split=0.2,
        seed=42
    )
    print(f"Training complete. Status: {ckpt_path}")





if __name__ == "__main__":
    main()
