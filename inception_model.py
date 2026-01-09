
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as L
from typing import List, Tuple, Optional, Any
import numpy as np
from cnn_model import RadarDataModule, RadarLightningModule

# =========================
# 1) Inception-v3 Adapted Model
# =========================

class RadarInception(nn.Module):
    """
    Adapts a pre-trained Inception-v3 for 1-channel radar data.
    Note: Inception expects 299x299 input.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # 1. 1-channel to 3-channel adapter
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # 2. Load Inception-v3 Backbone
        # We set aux_logits=True by default for better convergence in training
        try:
            from torchvision.models import Inception_V3_Weights
            self.inception = models.inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
                aux_logits=True
            )
        except ImportError:
            self.inception = models.inception_v3(pretrained=pretrained, aux_logits=True)
        
        # 3. Modify Primary Output Head
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)
        
        # 4. Modify Auxiliary Output Head
        if self.inception.aux_logits:
            num_ftrs_aux = self.inception.AuxLogits.fc.in_features
            self.inception.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, H, W) -> e.g., (batch, 1, 50, 1280)
        
        # 1. Expand channels
        x = self.input_adapter(x)
        
        # 2. Resize to 299x299 (Inception standard)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 3. Pass through Inception
        # During training, inception returns (output, aux_output)
        # During eval, it returns only output
        return self.inception(x)

# =========================
# 2) Lightning Module
# =========================

class RadarInceptionLightningModule(RadarLightningModule):
    """
    Inherits metrics and logic from RadarLightningModule.
    Handles Inception's unique dual-output training step.
    """
    def __init__(self, num_classes: int, lr: float = 1e-4, weights: Optional[torch.Tensor] = None):
        super(RadarLightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = RadarInception(num_classes=num_classes, pretrained=True)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.num_classes = num_classes
        
        # Re-initialize metrics
        from torchmetrics.classification import Accuracy, ConfusionMatrix
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Inception training returns (logits, aux_logits)
        outputs = self(x)
        
        if self.training and self.model.inception.aux_logits:
            logits, aux_logits = outputs
            loss1 = self.criterion(logits, y)
            loss2 = self.criterion(aux_logits, y)
            loss = loss1 + 0.4 * loss2 # Standard Inception weighting
        else:
            logits = outputs
            loss = self.criterion(logits, y)
            
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

# =========================
# 3) Entry Point
# =========================

def run_inception_training(
    X: List[np.ndarray],
    y: List[int],
    batch_size: int = 8, # Inception is memory heavy, smaller batch size recommended
    lr: float = 1e-4,
    epochs: int = 20,
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
    model = RadarInceptionLightningModule(num_classes=num_classes, lr=lr, weights=weights)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="mps",
        devices=1,
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False
    )
    
    print("\n--- Starting Pre-trained Inception-v3 Training ---")
    trainer.fit(model, datamodule=datamodule)
    
    print("\n--- Running Testing phase ---")
    trainer.test(model, datamodule=datamodule)

    # Export to .pth
    save_path = "radar_inception.pth"
    torch.save({
        "model_state": model.model.state_dict(),
        "mean": getattr(datamodule, "mean", 0.0),
        "std": getattr(datamodule, "std", 1.0),
        "num_classes": num_classes,
    }, save_path)
    
    return save_path
