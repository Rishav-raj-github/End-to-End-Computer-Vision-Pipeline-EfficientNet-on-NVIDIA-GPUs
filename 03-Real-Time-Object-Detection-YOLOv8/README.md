# 03-Real-Time-Object-Detection-YOLOv8

## Summary
This module implements real-time object detection using YOLOv8 (You Only Look Once version 8), optimized for NVIDIA GPUs. It provides high-performance inference capabilities with support for model export and deployment on edge devices and production environments.

## Features

### YOLOv8 Architecture
- State-of-the-art object detection with improved accuracy and speed
- Multiple model variants (nano, small, medium, large, extra-large)
- Support for various input resolutions
- Multi-scale feature detection

### NVIDIA GPU Optimization
- CUDA-accelerated inference for maximum throughput
- TensorRT integration for optimized GPU performance
- Mixed precision (FP16/FP32) support
- Batch processing capabilities

### ONNX Export
- Cross-platform model portability
- Framework-agnostic deployment
- Optimized inference graphs
- Support for various ONNX runtimes

### Triton Inference Server Deployment
- Scalable model serving infrastructure
- Dynamic batching for improved throughput
- Multi-model ensemble support
- REST and gRPC API endpoints
- Production-ready deployment pipeline

## Installation

```bash
# Install required dependencies
pip install ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime-gpu
pip install tritonclient[all]

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Demo Commands

### Basic Inference
```bash
# Run detection on image
python detect.py --source image.jpg --weights yolov8n.pt

# Run detection on video
python detect.py --source video.mp4 --weights yolov8s.pt

# Real-time webcam detection
python detect.py --source 0 --weights yolov8m.pt
```

### GPU-Accelerated Inference
```bash
# Single GPU inference
python detect.py --source data/ --device 0 --weights yolov8l.pt

# Multi-GPU inference
python detect.py --source data/ --device 0,1 --weights yolov8x.pt
```

### Model Export
```bash
# Export to ONNX
python export_model.py --weights yolov8n.pt --format onnx

# Export to TensorRT
python export_model.py --weights yolov8n.pt --format engine --device 0
```

### Triton Deployment
```bash
# Prepare model repository
python prepare_triton.py --model yolov8n.pt --output triton_models/

# Launch Triton server
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:latest tritonserver --model-repository=/models

# Send inference request
python triton_client.py --image test.jpg --url localhost:8000
```

## Future Extensions

- **Custom Dataset Training**: Fine-tuning YOLOv8 on domain-specific datasets
- **Multi-Camera Fusion**: Synchronized detection across multiple camera streams
- **Object Tracking**: Integration with SORT/DeepSORT for persistent object tracking
- **Edge Deployment**: Optimization for NVIDIA Jetson devices (Nano, Xavier, Orin)
- **Augmented Reality**: Real-time object detection with AR visualization
- **Cloud Scaling**: Kubernetes-based auto-scaling for high-volume workloads
- **Quantization**: INT8 quantization for improved inference speed
- **Active Learning**: Continuous model improvement with human-in-the-loop feedback
