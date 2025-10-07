# Vision Transformers & EfficientNet Pipeline 2025 | GPU-Accelerated Deep Learning | End-to-End Computer Vision

A modern, production-grade computer vision pipeline that unifies EfficientNet and Vision Transformers (ViT) for state-of-the-art image understanding on NVIDIA GPUs. Built for research-to-production workflows with PyTorch/TensorFlow training, FP16/Tensor Cores acceleration, ONNX export, and Triton Inference Server deployment at scale.

---

## Badges

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

## Professional Overview

This repository provides an end-to-end pipeline covering data ingestion, training (EfficientNet/ViT), evaluation, conversion, and scalable deployment. It is engineered for reproducibility, performance, and extensibility:

- Hybrid backbones: EfficientNet (B0–B7) and ViT (Base/Large) with interchangeable heads.
- Mixed precision training (AMP/FP16) with automatic gradient scaling on NVIDIA GPUs.
- Efficient data input pipeline with heavy augmentations (RandAugment, Mixup/CutMix, AutoAugment).
- Export to ONNX/TensorRT and serve via Triton Inference Server with model repository structure.
- CI-ready config-driven experiments with YAML/JSON, experiment logging, and checkpointing.

---

## 2025 Roadmap — 5 Advanced Modules

1) State-of-the-Art Classification
   - Implement EfficientNet and ViT baselines with SOTA augmentations and schedulers (OneCycle, Cosine, Warmup).
   - Add knowledge distillation and label smoothing.
   - Support multi-GPU DistributedDataParallel.

2) Object Detection
   - Integrate EfficientDet/ViTDet adapters or Detic-style heads.
   - COCO-style evaluation with mAP metrics; export to ONNX/TensorRT.
   - Real-time inference demos with torchvision/ultralytics adapters.

3) Transfer Learning & Fine-Tuning
   - Frozen backbone, linear probe, and full fine-tune recipes.
   - Hyperparameter search templates (Optuna/Ray Tune).
   - Domain adaptation tips and few-shot protocols.

4) Scalable Deployment
   - Triton model repository with ensemble graphs (preprocess -> model -> postprocess).
   - GPU batching, dynamic shape support, and A/B versions.
   - Containerized deployment with Docker Compose and optional Kubernetes/Helm.

5) Visual Explainability
   - Grad-CAM/Score-CAM and attention rollout for ViT.
   - Saliency maps and misclassification analysis dashboards.
   - Model cards and governance notes for responsible AI.

---

## Quickstart

- Clone and set up environment
  - conda create -n cv-pipeline python=3.11 -y
  - conda activate cv-pipeline
  - pip install -r requirements.txt

- Train EfficientNet/ViT (example)
  - python train.py --model efficientnet_b0 --data data/imagenet_sample --epochs 50 --amp
  - python train.py --model vit_base --data data/imagenet_sample --epochs 50 --amp

- Export and serve with Triton
  - python export_onnx.py --model checkpoints/best.pt --out models/model.onnx
  - docker compose up -d triton

---

## Repository Structure (planned)

- 01-medium-advanced-projects/
  - 01-vision-transformer-classification/
    - README.md (Module 1 blueprint)
- src/
  - data, models, train, eval, export, serving
- configs/
  - experiment YAMLs
- notebooks/
  - exploration and visualization

---

## Contributing

Contributions welcome! Please open issues/PRs for enhancements, bugfixes, and docs.

## License

MIT License — see LICENSE for details.
