
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as L
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Any
from cnn_model import RadarDataModule, RadarLightningModule, RadarDataset

# =========================
# 1) ResNet-18 Adapted Model
# =========================

class RadarResNet(nn.Module):
    """
    Adapts a pre-trained ResNet-18 for 1-channel radar data.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # 1. 1-channel to 3-channel adapter
        # ResNet expects 3 channels (RGB). We use a 1x1 conv to expand our 1 channel.
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # 2. Resizer
        # Range-Time Maps are (50, 1280). ResNet expects square images (usually 224x224).
        # We will resize using interpolation in the forward pass.
        
        # 3. Load ResNet-18 Backbone
        # Note: In newer torchvision, weights are used instead of 'pretrained=True'
        try:
            from torchvision.models import ResNet18_Weights
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except ImportError:
            # Fallback for older torchvision
            self.resnet = models.resnet18(pretrained=pretrained)
        
        # 4. Modify Output Head
        # ResNet-18's final layer is 'fc'. We replace it with a layer matching our classes.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 50, 1280)
        
        # 1. Expand channels: (batch, 1, 50, 1280) -> (batch, 3, 50, 1280)
        x = self.input_adapter(x)
        
        # 2. Resize to 224x224 for ResNet
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 3. Pass through ResNet
        return self.resnet(x)

# =========================
# 2) Lightning Module
# =========================

class RadarPretrainedLightningModule(RadarLightningModule):
    """
    Inherits metrics and steps from the base module, but uses ResNet backbone.
    """
    def __init__(self, num_classes: int, lr: float = 1e-4, weights: Optional[torch.Tensor] = None):
        super(RadarLightningModule, self).__init__() # Skip RadarLightningModule's init
        self.save_hyperparameters()
        self.model = RadarResNet(num_classes=num_classes, pretrained=True)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.num_classes = num_classes
        
        # Re-import metrics initialization from base class logic
        from torchmetrics.classification import Accuracy, ConfusionMatrix
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

# =========================
# 3) Specialized Training Entry
# =========================

import numpy as np

def run_pretrained_training(
    X: List[np.ndarray],
    y: List[int],
    batch_size: int = 16,
    lr: float = 1e-4, # Pretrained models usually prefer smaller learning rates
    epochs: int = 50,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 4
) -> str:
    L.seed_everything(seed)
    
    num_classes = len(set(y))
    
    from collections import Counter
    counts = Counter(y)
    total = sum(counts.values())
    weights = torch.tensor([total / (len(counts) * counts[i]) for i in range(num_classes)], dtype=torch.float)
    
    datamodule = RadarDataModule(X, y, batch_size=batch_size, val_split=val_split, test_split=test_split, seed=seed, num_workers=num_workers)
    model = RadarPretrainedLightningModule(num_classes=num_classes, lr=lr, weights=weights)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="mps", # Optimized for your Mac
        devices=1,
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False
    )
    
    print("\n--- Starting Pre-trained ResNet-18 Training ---")
    trainer.fit(model, datamodule=datamodule)
    
    print("\n--- Running Testing phase ---")
    trainer.test(model, datamodule=datamodule)

    # Export to .pth
    save_path = "radar_resnet.pth"
    torch.save({
        "model_state": model.model.state_dict(),
        "mean": getattr(datamodule, "mean", 0.0),
        "std": getattr(datamodule, "std", 1.0),
        "num_classes": num_classes,
    }, save_path)
        
    return save_path
