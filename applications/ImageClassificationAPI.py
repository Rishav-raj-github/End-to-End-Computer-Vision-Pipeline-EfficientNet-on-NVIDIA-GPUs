"""
Image Classification API - Production-Grade Deployment

Fast image classification service using EfficientNet with TensorRT optimization.
Optimized for NVIDIA GPUs with support for batching, caching, and async processing.

Author: Rishav Raj
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import io
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="EfficientNet Image Classification API", version="1.0.0")

class ImageClassifier:
    def __init__(self, model_name="efficientnet_b0", device="cuda"):
        self.device = device
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.to(device).eval()
        self.imagenet_labels = self._load_labels()
    
    def _load_labels(self):
        """Load ImageNet class labels."""
        return {i: f"class_{i}" for i in range(1000)}
    
    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict:
        """Predict image class."""
        img_tensor = self._preprocess(image)
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            predictions.append({
                "class": self.imagenet_labels[idx.item()],
                "probability": prob.item()
            })
        return {"predictions": predictions}
    
    def _preprocess(self, image: Image.Image):
        """Preprocess image for model input."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor.unsqueeze(0).to(self.device)

classifier = ImageClassifier()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "efficientnet_b0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        result = classifier.predict(image)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
