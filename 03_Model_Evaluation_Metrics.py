"""
Model Evaluation and Performance Metrics

Comprehensive evaluation framework including accuracy, precision, recall, F1, AUC metrics.

Author: Rishav Raj
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np
from typing import Dict, Tuple


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, dataloader, num_classes: int) -> Dict:
        """Complete evaluation on dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return self._compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
    
    def _compute_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        return metrics


if __name__ == "__main__":
    print("Model evaluation framework ready")
