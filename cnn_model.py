
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from torchmetrics.classification import Accuracy, ConfusionMatrix
from typing import List, Tuple, Optional, Any

# =========================
# 1) Dataset & DataModule
# =========================

class RadarDataset(Dataset):
    """
    Wraps radar samples into a PyTorch Dataset.
    """
    def __init__(self, X: List[np.ndarray], y: List[int], mean: float = 0.0, std: float = 1.0):
        self.X = []
        self.y = [int(v) for v in y]
        
        for arr in X:
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]    # (1, 200, 1280)
            elif arr.ndim == 3 and arr.shape[0] != 1:
                if arr.shape[-1] == 1:
                    arr = np.transpose(arr, (2, 0, 1))
                else:
                    raise ValueError(f"Unexpected shape {arr.shape}.")
            self.X.append(arr.astype(np.float32))
            
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        x = (x - self.mean) / (self.std + 1e-8)
        y = torch.tensor(self.y[idx]).long()
        return x, y

class RadarDataModule(L.LightningDataModule):
    def __init__(self, X: List[np.ndarray], y: List[int], batch_size: int = 16, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42, num_workers: int = 0):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        if len(self.X) == 0:
            return
            
        # Calculate global mean/std for normalization
        idx = np.random.choice(len(self.X), size=min(128, len(self.X)), replace=False)
        subset = np.stack([self.X[i] for i in idx])
        self.mean = subset.mean()
        self.std = subset.std()

        # Split: Train, Val, Test
        full_dataset = RadarDataset(self.X, self.y, mean=self.mean, std=self.std)
        total_len = len(full_dataset)
        n_test = int(total_len * self.test_split)
        n_val = int(total_len * self.val_split)
        n_train = total_len - n_test - n_val
        
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_val, n_test], 
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

# =========================
# 2) Lightning Model
# =========================

class RadarCNN(nn.Module):
    """
    Original CNN architecture.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(11520, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, 1)
        return self.classifier(z)

class RadarLightningModule(L.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = RadarCNN(num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.num_classes = num_classes
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Class-wise Metrics
        self.val_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        
        # Confusion Matrix
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.val_class_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_acc(logits, y)
        self.test_class_acc(logits, y)
        self.test_cm(logits, y)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc, sync_dist=True)
        return loss

    def on_test_epoch_end(self) -> None:
        # Final Summary Reporting
        print("\n" + "="*50)
        print("FINAL TEST RESULTS SUMMARY")
        print("="*50)
        
        overall_acc = self.test_acc.compute()
        print(f"Overall Test Accuracy: {overall_acc:.4f}")
        
        print("\nClass-wise Accuracy:")
        class_accs = self.test_class_acc.compute()
        for i, acc in enumerate(class_accs):
            print(f"  - Class {i}: {acc:.4f}")
            
        print("\nConfusion Matrix:")
        cm = self.test_cm.compute().cpu().numpy()
        print(cm)
        print("="*50 + "\n")
        
        # Log to tensorboard or other loggers if active
        for i, acc in enumerate(class_accs):
            self.log(f"test_acc_class_{i}", acc)

    def on_validation_epoch_end(self) -> None:
        # Log class-wise accuracy for monitoring
        accuracies = self.val_class_acc.compute()
        # You can print here too if you want to see class accuracy during training
        self.val_class_acc.reset()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

# =========================
# 3) Entry Point
# =========================

def run_training(
    X: List[np.ndarray],
    y: List[int],
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 20,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 4
) -> str:
    L.seed_everything(seed)
    
    num_classes = len(set(y))
    
    # Calculate class weights for better performance on hard classes
    from collections import Counter
    counts = Counter(y)
    total = sum(counts.values())
    weights = torch.tensor([total / (len(counts) * counts[i]) for i in range(num_classes)], dtype=torch.float)
    
    datamodule = RadarDataModule(X, y, batch_size=batch_size, val_split=val_split, test_split=test_split, seed=seed, num_workers=num_workers)
    model = RadarLightningModule(num_classes=num_classes, lr=lr, weights=weights)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="mps",
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    
    # 1. Training & Validation
    trainer.fit(model, datamodule=datamodule)
    
    # 2. Testing (uses current weights since checkpointing is off)
    print("\nRunning Testing phase...")
    trainer.test(model, datamodule=datamodule)

    # 3. Export to .pth (Space efficient)
    save_path = "radar_cnn.pth"
    torch.save({
        "model_state": model.model.state_dict(),
        "mean": getattr(datamodule, "mean", 0.0),
        "std": getattr(datamodule, "std", 1.0),
        "num_classes": num_classes,
    }, save_path)
        
    return save_path

def load_model_for_inference(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, float, float, int, torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt["num_classes"]
    model = RadarCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("mean", 0.0), ckpt.get("std", 1.0), num_classes, device