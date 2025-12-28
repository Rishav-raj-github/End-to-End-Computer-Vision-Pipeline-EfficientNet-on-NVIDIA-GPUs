"""
EfficientNet Training Pipeline on NVIDIA GPUs

Comprehensive training pipeline with mixed precision (FP16), gradient accumulation,
and learning rate scheduling optimized for Tesla V100/A100 GPUs.

Author: Rishav Raj
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EfficientNetTrainer:
    """Training manager for EfficientNet models."""
    
    def __init__(
        self,
        model_variant: str = "b0",
        num_classes: int = 1000,
        learning_rate: float = 0.016,
        device: str = "cuda"
    ):
        self.device = device
        self.model = self._load_model(model_variant, num_classes)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
        self.scaler = GradScaler()  # FP16 mixed precision
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def _load_model(self, variant: str, num_classes: int):
        """Load EfficientNet variant."""
        model_dict = {
            "b0": models.efficientnet_b0,
            "b1": models.efficientnet_b1,
            "b2": models.efficientnet_b2,
            "b3": models.efficientnet_b3,
            "b4": models.efficientnet_b4,
            "b5": models.efficientnet_b5,
        }
        model = model_dict[variant](pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model.to(self.device)
    
    def train_epoch(
        self,
        train_loader,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model performance."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, best_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": best_acc
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")


class LearningRateScheduler:
    """Cosine annealing with warm restarts."""
    
    def __init__(
        self,
        optimizer,
        base_lr: float = 0.016,
        max_epochs: int = 350,
        warmup_epochs: int = 5
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
    
    def step(self, epoch: int):
        """Update learning rate for current epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)))
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        return lr


if __name__ == "__main__":
    trainer = EfficientNetTrainer(model_variant="b0", num_classes=1000)
    print(f"Model initialized on {trainer.device}")
    print(f"Total parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
