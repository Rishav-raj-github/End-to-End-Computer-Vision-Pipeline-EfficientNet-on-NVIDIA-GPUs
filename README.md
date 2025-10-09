# 🚀 End-to-End Computer Vision Pipeline | EfficientNet & Vision Transformers on NVIDIA GPUs

> **A modern, production-grade computer vision pipeline** that unifies EfficientNet and Vision Transformers (ViT) for state-of-the-art image understanding on NVIDIA GPUs. Built for research-to-production workflows with PyTorch/TensorFlow training, FP16/Tensor Cores acceleration, ONNX export, and Triton Inference Server deployment at scale.

---

## 📊 Badges

![EfficientNet](https://img.shields.io/badge/Model-EfficientNet-B2F7EF?logo=readme&logoColor=white)
![Vision Transformer](https://img.shields.io/badge/Model-ViT-7B1FA2)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/Format-ONNX-005CED?logo=onnx&logoColor=white)
![Triton](https://img.shields.io/badge/Serving-NVIDIA%20Triton-76B900?logo=nvidia&logoColor=white)
![CUDA FP16](https://img.shields.io/badge/Accelerator-CUDA%20FP16-76B900)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)
![MIT](https://img.shields.io/badge/License-MIT-black)

---

## 🎯 Professional Overview

This repository provides an **end-to-end pipeline** covering data ingestion, training (EfficientNet/ViT), evaluation, conversion, and scalable deployment. It is engineered for reproducibility, performance, and extensibility:

- ✅ **Hybrid backbones**: EfficientNet (B0–B7) and ViT (Base/Large) with interchangeable heads
- ✅ **Mixed precision training** (AMP/FP16) with automatic gradient scaling on NVIDIA GPUs
- ✅ **Efficient data pipeline** with advanced augmentations (RandAugment, Mixup/CutMix, AutoAugment)
- ✅ **Export to ONNX/TensorRT** and serve via Triton Inference Server with model repository structure
- ✅ **CI-ready config-driven experiments** with YAML/JSON, experiment logging, and checkpointing

---

## 🗺️ Professional Project Roadmap (2025)

### **Active Modules**

#### 🟢 [01-Vision-Transformer-Classification](./01-medium-advanced-projects/01-vision-transformer-classification/)
- **Status**: In Development
- **Description**: State-of-the-art image classification with EfficientNet and Vision Transformers
- **Technologies**: PyTorch, timm, NVIDIA CUDA, Tensor Cores
- **Features**: Multi-GPU training, SOTA augmentations, knowledge distillation, label smoothing

#### 🟡 [02-Self-Supervised-Contrastive-Learning](./02-Self-Supervised-Contrastive-Learning/)
- **Status**: Planned
- **Description**: Advanced self-supervised image representation learning using SimCLR, BYOL, and MoCo
- **Technologies**: PyTorch, NVIDIA GPU optimization, distributed training
- **Features**: Contrastive learning frameworks, large batch training, linear evaluation protocols

#### 🟡 [03-Real-Time-Object-Detection-YOLOv8](./03-Real-Time-Object-Detection-YOLOv8/)
- **Status**: Planned
- **Description**: Real-time object detection pipeline with YOLOv8 optimized for NVIDIA GPUs
- **Technologies**: Ultralytics YOLOv8, TensorRT, ONNX Runtime
- **Features**: End-to-end training, ONNX/TensorRT export, real-time inference optimization

#### 🟡 [04-Image-Segmentation-Transformers](./04-Image-Segmentation-Transformers/)
- **Status**: Planned
- **Description**: Advanced semantic segmentation with Vision Transformers (ViT, Segmenter)
- **Technologies**: PyTorch, Segmentation Models, TensorRT
- **Features**: Transfer learning, GPU parallelization, multi-scale inference, deployment pipelines

### **Upgrade Notes**

- **Q1 2025**: Complete Module 01 with full documentation and benchmarks
- **Q2 2025**: Launch Module 02 (Self-Supervised Learning) with SimCLR implementation
- **Q3 2025**: Deploy Modules 03 & 04 (Object Detection & Segmentation)
- **Q4 2025**: Add federated learning and privacy-preserving training modules

---

## ✅ How to Run & Train: Complete Checklist

### **1️⃣ Environment Setup**

```bash
# Clone the repository
git clone https://github.com/Rishav-raj-github/End-to-End-Computer-Vision-Pipeline-EfficientNet-on-NVIDIA-GPUs.git
cd End-to-End-Computer-Vision-Pipeline-EfficientNet-on-NVIDIA-GPUs

# Create and activate conda environment
conda create -n cv-pipeline python=3.11 -y
conda activate cv-pipeline

# Install dependencies
pip install -r requirements.txt

# Verify CUDA and GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

### **2️⃣ Data Preparation**

```bash
# Download and prepare ImageNet or custom dataset
python scripts/prepare_data.py --dataset imagenet --output data/imagenet

# Apply data augmentation preview
python scripts/visualize_augmentations.py --config configs/augmentation.yaml
```

### **3️⃣ Training**

```bash
# Train EfficientNet-B0 with mixed precision
python train.py \
  --model efficientnet_b0 \
  --data data/imagenet \
  --epochs 100 \
  --batch-size 128 \
  --amp \
  --gpu 0,1,2,3

# Train Vision Transformer (ViT-Base)
python train.py \
  --model vit_base_patch16_224 \
  --data data/imagenet \
  --epochs 100 \
  --batch-size 256 \
  --amp \
  --distributed

# Resume training from checkpoint
python train.py \
  --model efficientnet_b3 \
  --resume checkpoints/efficientnet_b3_epoch50.pth \
  --epochs 100
```

### **4️⃣ Model Export**

```bash
# Export to ONNX format
python export_onnx.py \
  --model checkpoints/best_model.pth \
  --output models/model.onnx \
  --opset 14 \
  --simplify

# Convert to TensorRT engine
python export_tensorrt.py \
  --onnx models/model.onnx \
  --output models/model.trt \
  --fp16 \
  --workspace 4096

# Validate exported model
python validate_export.py \
  --pytorch checkpoints/best_model.pth \
  --onnx models/model.onnx \
  --trt models/model.trt
```

### **5️⃣ Deployment with Triton Inference Server**

```bash
# Prepare Triton model repository
python scripts/prepare_triton_repo.py \
  --model models/model.onnx \
  --output triton-models/ \
  --config configs/triton_config.pbtxt

# Start Triton Inference Server
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton-models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Run inference client
python client.py \
  --triton-url localhost:8000 \
  --model-name efficientnet_b0 \
  --image test_images/sample.jpg
```

### **6️⃣ Evaluation & Monitoring**

```bash
# Evaluate model on validation set
python evaluate.py \
  --model checkpoints/best_model.pth \
  --data data/imagenet/val \
  --metrics accuracy,top5,loss

# Generate confusion matrix and metrics
python scripts/generate_metrics.py \
  --predictions outputs/predictions.json \
  --labels data/labels.json

# Launch TensorBoard for training visualization
tensorboard --logdir=logs/tensorboard --port=6006
```

---

## 📂 Repository Structure

```
.
├── 01-medium-advanced-projects/
│   └── 01-vision-transformer-classification/
│       ├── README.md                    # Module 1 documentation
│       ├── train.py                     # Training script
│       ├── model.py                     # Model architectures
│       └── data.py                      # Data loaders
├── 02-Self-Supervised-Contrastive-Learning/
│   └── README.md                        # Self-supervised learning docs
├── 03-Real-Time-Object-Detection-YOLOv8/
│   └── README.md                        # YOLOv8 detection pipeline
├── 04-Image-Segmentation-Transformers/
│   └── README.md                        # Transformer segmentation
├── src/
│   ├── data/                            # Data loading and augmentation
│   ├── models/                          # Model definitions
│   ├── train/                           # Training utilities
│   ├── eval/                            # Evaluation scripts
│   ├── export/                          # Model export (ONNX, TRT)
│   └── serving/                         # Triton serving configs
├── configs/
│   ├── efficientnet_b0.yaml
│   ├── vit_base.yaml
│   └── augmentation.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── prepare_triton_repo.py
│   └── generate_metrics.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### **How to Contribute**

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure code quality:
   ```bash
   # Run linting
   flake8 src/ --max-line-length=120
   
   # Run tests
   pytest tests/ -v
   ```

3. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Description of your feature"
   ```

4. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### **Contribution Guidelines**

- ✅ Follow PEP 8 style guidelines for Python code
- ✅ Add docstrings to all functions and classes
- ✅ Include unit tests for new features
- ✅ Update documentation and README as needed
- ✅ Ensure all tests pass before submitting PR

### **Areas for Contribution**

- 🔧 Bug fixes and performance improvements
- 📚 Documentation enhancements
- 🚀 New model architectures and optimizations
- 🧪 Additional test coverage
- 🎨 Visualization tools and notebooks

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NVIDIA** for CUDA, TensorRT, and Triton Inference Server
- **PyTorch** and **TensorFlow** teams for deep learning frameworks
- **timm** library for pretrained vision models
- **Hugging Face** for Transformers and model hubs

---

## 📧 Contact & Support

For questions, issues, or collaboration opportunities:

- 🐛 **Issues**: [GitHub Issues](https://github.com/Rishav-raj-github/End-to-End-Computer-Vision-Pipeline-EfficientNet-on-NVIDIA-GPUs/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Rishav-raj-github/End-to-End-Computer-Vision-Pipeline-EfficientNet-on-NVIDIA-GPUs/discussions)
- 📧 **Email**: Contact via GitHub profile

---

**⭐ If you find this project helpful, please consider giving it a star!**
