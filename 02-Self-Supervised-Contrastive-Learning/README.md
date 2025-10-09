# 🧠 Self-Supervised Contrastive Learning for Image Representation

> **Advanced self-supervised learning pipeline** for learning robust visual representations without labels using state-of-the-art contrastive learning frameworks (SimCLR, BYOL, MoCo) optimized for NVIDIA GPUs.

---

## 📋 Overview

This module implements **self-supervised contrastive learning** techniques that enable models to learn powerful image representations from unlabeled data. These pre-trained representations can be fine-tuned for downstream tasks with minimal labeled data, achieving performance comparable to fully supervised methods.

### **Key Features**

- ✅ **Multiple Frameworks**: SimCLR, BYOL (Bootstrap Your Own Latent), and MoCo (Momentum Contrast)
- ✅ **NVIDIA GPU Optimization**: Multi-GPU training with mixed precision (FP16/AMP)
- ✅ **Large Batch Training**: Distributed Data Parallel (DDP) for batch sizes up to 4096+
- ✅ **Advanced Augmentations**: Strong augmentation pipelines (color jittering, Gaussian blur, random crop)
- ✅ **Linear Evaluation**: Standard protocols for evaluating learned representations
- ✅ **Transfer Learning**: Fine-tuning pipelines for various downstream tasks

---

## 🎯 Use Cases

1. **Medical Imaging**: Learn representations from unlabeled medical scans, fine-tune on small labeled datasets
2. **Satellite Imagery**: Pre-train on massive unlabeled satellite data for downstream classification/segmentation
3. **Industrial Inspection**: Learn defect patterns without extensive labeling
4. **General Computer Vision**: Create powerful backbone models for transfer learning

---

## 🔬 Core Technologies

### **Frameworks & Libraries**

- **PyTorch**: Deep learning framework with native GPU acceleration
- **PyTorch Lightning**: High-level training framework for scalable experiments
- **NVIDIA CUDA**: GPU acceleration with Tensor Cores
- **NVIDIA Apex**: Mixed precision training utilities
- **Distributed Data Parallel (DDP)**: Multi-GPU synchronous training

### **Contrastive Learning Methods**

#### **1. SimCLR (Simple Framework for Contrastive Learning)**
- Large batch sizes (2048-8192) with gradient accumulation
- Temperature-scaled NT-Xent loss
- Strong data augmentation pipeline
- Projection head with 2-3 layer MLP

#### **2. BYOL (Bootstrap Your Own Latent)**
- No negative pairs required
- Online and target networks with exponential moving average (EMA)
- Predictor network for asymmetric architecture
- More stable training without temperature tuning

#### **3. MoCo (Momentum Contrast)**
- Queue-based memory bank for large number of negative samples
- Momentum encoder for consistent representations
- Efficient memory usage with dynamic queue
- MoCo v2/v3 improvements with MLP projection head

---

## 🚀 Sample Workflow

### **Phase 1: Self-Supervised Pre-training**

```bash
# SimCLR pre-training on ImageNet (unlabeled)
python pretrain_simclr.py \
  --backbone resnet50 \
  --data data/imagenet_unlabeled \
  --batch-size 512 \
  --epochs 800 \
  --temperature 0.5 \
  --projection-dim 128 \
  --gpus 0,1,2,3,4,5,6,7 \
  --distributed \
  --amp

# BYOL pre-training (no negative pairs)
python pretrain_byol.py \
  --backbone resnet50 \
  --data data/imagenet_unlabeled \
  --batch-size 256 \
  --epochs 1000 \
  --ema-decay 0.996 \
  --gpus 0,1,2,3 \
  --distributed

# MoCo pre-training with queue
python pretrain_moco.py \
  --backbone resnet50 \
  --data data/imagenet_unlabeled \
  --batch-size 256 \
  --epochs 800 \
  --queue-size 65536 \
  --momentum 0.999 \
  --gpus 0,1,2,3,4,5,6,7
```

### **Phase 2: Linear Evaluation**

```bash
# Freeze backbone, train linear classifier
python linear_eval.py \
  --pretrained checkpoints/simclr_resnet50_800ep.pth \
  --data data/imagenet \
  --batch-size 256 \
  --epochs 100 \
  --lr 0.1 \
  --freeze-backbone
```

### **Phase 3: Fine-tuning for Downstream Tasks**

```bash
# Full fine-tuning on downstream task
python finetune.py \
  --pretrained checkpoints/simclr_resnet50_800ep.pth \
  --data data/custom_dataset \
  --batch-size 128 \
  --epochs 50 \
  --lr 0.001 \
  --unfreeze-after-epoch 10
```

### **Phase 4: Export & Deployment**

```bash
# Export fine-tuned model to ONNX
python export_onnx.py \
  --model checkpoints/finetuned_model.pth \
  --output models/model.onnx

# Convert to TensorRT for inference
python export_tensorrt.py \
  --onnx models/model.onnx \
  --output models/model.trt \
  --fp16
```

---

## 📂 Planned Code Structure

```
02-Self-Supervised-Contrastive-Learning/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── configs/
│   ├── simclr_config.yaml             # SimCLR hyperparameters
│   ├── byol_config.yaml               # BYOL hyperparameters
│   └── moco_config.yaml               # MoCo hyperparameters
├── src/
│   ├── models/
│   │   ├── simclr.py                  # SimCLR implementation
│   │   ├── byol.py                    # BYOL implementation
│   │   ├── moco.py                    # MoCo implementation
│   │   └── backbones.py               # ResNet, ViT backbones
│   ├── data/
│   │   ├── dataset.py                 # Custom dataset loaders
│   │   ├── augmentations.py           # Contrastive augmentations
│   │   └── transforms.py              # Standard transforms
│   ├── losses/
│   │   ├── nt_xent.py                 # NT-Xent loss (SimCLR)
│   │   ├── byol_loss.py               # BYOL loss
│   │   └── moco_loss.py               # MoCo loss
│   └── utils/
│       ├── logging.py                 # TensorBoard/Wandb logging
│       ├── metrics.py                 # Evaluation metrics
│       └── distributed.py             # DDP utilities
├── scripts/
│   ├── pretrain_simclr.py             # SimCLR pre-training script
│   ├── pretrain_byol.py               # BYOL pre-training script
│   ├── pretrain_moco.py               # MoCo pre-training script
│   ├── linear_eval.py                 # Linear evaluation
│   ├── finetune.py                    # Fine-tuning script
│   └── export_onnx.py                 # Model export
├── notebooks/
│   ├── 01_augmentation_visualization.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_representation_analysis.ipynb
└── tests/
    ├── test_models.py
    ├── test_losses.py
    └── test_data.py
```

---

## 🔧 NVIDIA GPU Optimization Strategies

### **1. Mixed Precision Training (FP16)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for images in dataloader:
    with autocast():
        loss = model(images)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### **2. Distributed Data Parallel (Multi-GPU)**
```bash
torchrun --nproc_per_node=8 pretrain_simclr.py --distributed
```

### **3. Gradient Accumulation for Large Batches**
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### **4. NVIDIA DALI for Data Loading**
- GPU-accelerated data augmentation
- Eliminates CPU bottlenecks
- 2-3x faster training throughput

---

## 📊 Expected Results

| Method | Backbone | Pre-train Epochs | ImageNet Top-1 (Linear Eval) | Notes |
|--------|----------|------------------|-------------------------------|-------|
| SimCLR | ResNet-50 | 800 | 69.3% | Batch size 4096 |
| BYOL | ResNet-50 | 1000 | 74.3% | No negative pairs |
| MoCo v3 | ResNet-50 | 800 | 73.8% | Queue size 65536 |
| SimCLR | ViT-B/16 | 300 | 75.5% | Transformer backbone |

---

## 🗓️ Future Plans & Roadmap

### **Q1 2025**
- ✅ Implement SimCLR with multi-GPU support
- ✅ Add BYOL and MoCo variants
- ✅ Benchmark on ImageNet

### **Q2 2025**
- 🔄 Add SwAV (Swapped Assignment Views)
- 🔄 Implement DINO (self-distillation with Vision Transformers)
- 🔄 Add VICReg (Variance-Invariance-Covariance Regularization)

### **Q3 2025**
- 🔄 Multi-modal contrastive learning (CLIP-style)
- 🔄 Video contrastive learning
- 🔄 Integration with downstream task benchmarks

### **Q4 2025**
- 🔄 AutoML for hyperparameter search
- 🔄 Deployment examples with Triton
- 🔄 Production-ready pipelines

---

## 📚 References

1. **SimCLR**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
2. **BYOL**: [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)
3. **MoCo**: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
4. **MoCo v2**: [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)
5. **MoCo v3**: [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)

---

## 🤝 Contributing

Contributions are welcome! Please see the main repository [Contributing Guide](../README.md#-contributing) for details.

---

## 📄 License

This project is licensed under the MIT License - see the main repository [LICENSE](../LICENSE) file.

---

**⭐ If you find this module helpful, please star the main repository!**
