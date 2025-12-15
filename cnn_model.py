from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# 1) Dataset
# =========================
class RadarDataset(Dataset):
    """
    Wraps radar samples into a PyTorch Dataset.

    Parameters:
        X: list of arrays (each (200, 1280) or (1, 200, 1280), dtype float32/float64)
        y: list of int labels, same length as X
        normalize: if True, standardize to (x - mean) / std using dataset-level stats
    """
    def __init__(self, X: List[np.ndarray], y: List[int], normalize: bool = True):
        assert len(X) == len(y), "X and y must have the same length"
        self.normalize = normalize
        self.y = [int(v) for v in y]
        self.X = []

        # Ensure channel-first (C,H,W) with C=1
        for arr in X:
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]    # (1, 200, 1280)
            elif arr.ndim == 3 and arr.shape[0] != 1:
                # If array is (H, W, 1), transpose to (1, H, W)
                if arr.shape[-1] == 1:
                    arr = np.transpose(arr, (2, 0, 1))
                else:
                    raise ValueError(f"Unexpected shape {arr.shape}. Expected (200,1280) or (1,200,1280).")
            self.X.append(arr.astype(np.float32))

        # Compute dataset-level normalization stats
        if self.normalize and len(self.X) > 0:
            idx = np.random.choice(len(self.X), size=min(512, len(self.X)), replace=False)
            sample = torch.from_numpy(np.stack([self.X[i] for i in idx], axis=0))  # (N,1,200,1280)
            self.mean = sample.mean().item()
            self.std = sample.std().item() + 1e-8
        else:
            self.mean, self.std = 0.0, 1.0

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])      # (1, 200, 1280)
        if self.normalize:
            x = (x - self.mean) / self.std
        y = torch.tensor(self.y[idx]).long()
        return x, y


# =========================
# 2) CNN Model
# =========================
class RadarCNN(nn.Module):
    """
    Compact 2D CNN for inputs (B, 1, 200, 1280).
    Pooling reduces width more aggressively than height,
    preserving slow-time resolution while reducing range bins.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # -> (16, 200, 1280)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),                     # -> (16, 100, 320)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),# -> (32, 100, 320)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),                     # -> (32, 50, 80)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),# -> (64, 50, 80)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),                     # -> (64, 25, 40)

            # Block 4
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),# -> (96, 25, 40)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),                     # -> (96, 25, 20)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(96 * 25 * 20, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        z = self.features(x)            # (B, 96, 25, 20)
        z = torch.flatten(z, 1)         # (B, 96*25*20)
        return self.classifier(z)       # logits (B, num_classes)


# =========================
# 3) Train/Eval utilities
# =========================
def _train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running = 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * xb.size(0)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

    return running / len(loader.dataset), accuracy_score(y_true, y_pred)


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running = 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running += loss.item() * xb.size(0)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    return running / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def run_training(
    X: List[np.ndarray],
    y: List[int],
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 20,
    val_split: float = 0.2,
    seed: int = 42,
    ckpt_path: str = "radar_cnn_best.pt"
) -> str:
    """
    Train RadarCNN on provided arrays/labels and save best checkpoint.

    Returns:
        Path to the best checkpoint (.pt).
    """
    import numpy as _np
    import torch as _torch
    import torch.utils.data as _data

    # Reproducibility
    random = _np.random.RandomState(seed)
    _torch.manual_seed(seed)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    # Dataset and split
    full_ds = RadarDataset(X, y, normalize=True)
    n_total = len(full_ds)
    n_val = int(round(val_split * n_total))
    n_train = n_total - n_val
    tr_ds, va_ds = _data.random_split(
        full_ds,
        [n_train, n_val],
        generator=_torch.Generator().manual_seed(seed)
    )

    # DataLoaders
    train_loader = _data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = _data.DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model/optim
    num_classes = len(set(y))
    model = RadarCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_report = ("", None)  # (text, cm)
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, y_true, y_pred = _evaluate(model, val_loader, criterion, device)
        print(f"Epoch {ep:02d} | Train {tr_loss:.4f}/{tr_acc:.4f} | Val {va_loss:.4f}/{va_acc:.4f}")

        # Save best
        if va_acc > best_acc:
            best_acc = va_acc
            # Store report at best
            report = classification_report(y_true, y_pred, digits=4)
            cm = confusion_matrix(y_true, y_pred)
            best_report = (report, cm)

            torch.save({
                "model_state": model.state_dict(),
                "mean": getattr(full_ds, "mean", 0.0),
                "std": getattr(full_ds, "std", 1.0),
                "num_classes": num_classes,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint to {ckpt_path}")

    print("\nBest validation report:")
    print(best_report[0])
    print("Confusion matrix:")
    print(best_report[1])

    return ckpt_path


# =========================
# 4) Inference utilities
# =========================
def load_model_for_inference(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, float, float, int, torch.device]:
    """
    Load a trained checkpoint for inference.
    Returns: (model, mean, std, num_classes, device)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt["num_classes"]
    model = RadarCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean, std = ckpt.get("mean", 0.0), ckpt.get("std", 1.0)
    return model, mean, std, num_classes, device


@torch.no_grad()
def predict_single(arr_200x1280: np.ndarray,
                   model: nn.Module,
                   mean: float,
                   std: float,
                   device: torch.device) -> Tuple[int, np.ndarray]:
    """
    Predict class for a single (200, 1280) array.
    Returns: (pred_class, probs ndarray of shape (num_classes,))
    """
    if arr_200x1280.ndim == 2:
        x = arr_200x1280[np.newaxis, np.newaxis, :, :]  # (1,1,200,1280)
    elif arr_200x1280.ndim == 3 and arr_200x1280.shape[0] == 1:
        x = arr_200x1280[np.newaxis, :, :, :]
    else:
        raise ValueError(f"Unexpected shape {arr_200x1280.shape}")

    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    xt = (xt - mean) / std
    logits = model(xt)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, probs
