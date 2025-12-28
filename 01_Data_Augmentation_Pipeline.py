"""
Data Augmentation Pipeline for Computer Vision

Comprehensive data augmentation strategies optimized for NVIDIA GPUs.
Includes: geometric transforms, color augmentations, mixing strategies (Mixup, CutMix).

Author: Rishav Raj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from typing import Tuple, Optional, List


class GeometricAugmentations:
    """Geometric transformation augmentations."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def random_rotation(self, image: torch.Tensor, degrees: float = 30) -> torch.Tensor:
        """Random rotation augmentation."""
        if torch.rand(1) < self.prob:
            angle = torch.randint(-int(degrees), int(degrees), (1,)).item()
            return TF.rotate(image, angle)
        return image
    
    def random_affine(self, image: torch.Tensor) -> torch.Tensor:
        """Random affine transformation."""
        if torch.rand(1) < self.prob:
            angle = torch.randint(-45, 45, (1,)).item()
            translate = (torch.rand(2) * 0.3).tolist()
            scale = (torch.rand(1) * 0.3 + 0.7).item()
            shear = (torch.rand(1) * 20).item()
            return TF.affine(image, angle, translate, scale, shear)
        return image
    
    def random_perspective(self, image: torch.Tensor, distortion_scale: float = 0.5) -> torch.Tensor:
        """Random perspective transformation."""
        if torch.rand(1) < self.prob:
            return TF.perspective(image, distortion_scale=distortion_scale)
        return image
    
    def random_crop(self, image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Random crop augmentation."""
        if torch.rand(1) < self.prob:
            return TF.center_crop(image, size)
        return image


class ColorAugmentations:
    """Color space augmentations."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def random_brightness(self, image: torch.Tensor, brightness: float = 0.3) -> torch.Tensor:
        """Random brightness adjustment."""
        if torch.rand(1) < self.prob:
            factor = torch.rand(1) * brightness + (1 - brightness / 2)
            return TF.adjust_brightness(image, factor.item())
        return image
    
    def random_contrast(self, image: torch.Tensor, contrast: float = 0.3) -> torch.Tensor:
        """Random contrast adjustment."""
        if torch.rand(1) < self.prob:
            factor = torch.rand(1) * contrast + (1 - contrast / 2)
            return TF.adjust_contrast(image, factor.item())
        return image
    
    def random_saturation(self, image: torch.Tensor, saturation: float = 0.3) -> torch.Tensor:
        """Random saturation adjustment."""
        if torch.rand(1) < self.prob:
            factor = torch.rand(1) * saturation + (1 - saturation / 2)
            return TF.adjust_saturation(image, factor.item())
        return image
    
    def random_hue(self, image: torch.Tensor, hue: float = 0.3) -> torch.Tensor:
        """Random hue adjustment."""
        if torch.rand(1) < self.prob:
            hue_factor = (torch.rand(1) - 0.5) * hue
            return TF.adjust_hue(image, hue_factor.item())
        return image
    
    def random_gaussian_blur(self, image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Random Gaussian blur."""
        if torch.rand(1) < self.prob:
            return TF.gaussian_blur(image, kernel_size)
        return image


class MixingAugmentations(nn.Module):
    """Mixing-based augmentations (Mixup, CutMix, MixupCutMix)."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def mixup(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup augmentation: linear interpolation of images and labels.
        """
        batch_size = images.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels
    
    def cutmix(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CutMix augmentation: mixing at the patch level.
        """
        batch_size, _, height, width = images.shape
        lam = np.random.beta(self.alpha, self.alpha)
        
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        index = torch.randperm(batch_size)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (height * width)
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return images, mixed_labels
    
    def mixup_cutmix(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combined Mixup + CutMix augmentation.
        """
        if torch.rand(1) < 0.5:
            return self.mixup(images, labels)
        else:
            return self.cutmix(images, labels)


class AugmentationPipeline:
    """Complete augmentation pipeline for training."""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.geometric = GeometricAugmentations(prob=0.7)
        self.color = ColorAugmentations(prob=0.7)
        self.mixing = MixingAugmentations(alpha=1.0)
    
    def get_train_transforms(self) -> transforms.Compose:
        """Get training augmentation pipeline."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_val_transforms(self) -> transforms.Compose:
        """Get validation augmentation pipeline (minimal)."""
        return transforms.Compose([
            transforms.Resize(int(self.image_size * 1.143)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    # Example usage
    aug = AugmentationPipeline(image_size=224)
    train_transform = aug.get_train_transforms()
    val_transform = aug.get_val_transforms()
    print(f"Train transforms: {train_transform}")
    print(f"Val transforms: {val_transform}")
