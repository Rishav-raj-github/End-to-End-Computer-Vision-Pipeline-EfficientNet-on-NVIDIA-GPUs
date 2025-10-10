#!/usr/bin/env python3
"""
YOLOv8 Real-Time Object Detection Inference Script
Supports webcam, video files, and image folders
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Import utilities from local module
from yolov8_utils import draw_boxes


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Real-Time Object Detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--source', type=str, default='0', help='Source: 0 for webcam, video file, or image folder')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], help='Inference image size [height, width]')
    parser.add_argument('--device', type=str, default='', help='Device: cuda:0 or cpu')
    parser.add_argument('--save', action='store_true', help='Save annotated results')
    parser.add_argument('--show', action='store_true', help='Display results in real-time')
    parser.add_argument('--output-dir', type=str, default='runs/detect', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum detections per image')
    parser.add_argument('--classes', type=int, nargs='+', help='Filter by class: --classes 0 1 2')
    parser.add_argument('--hide-labels', action='store_true', help='Hide labels in visualization')
    parser.add_argument('--hide-conf', action='store_true', help='Hide confidence scores')
    parser.add_argument('--line-thickness', type=int, default=2, help='Bounding box thickness')
    parser.add_argument('--benchmark', action='store_true', help='Print FPS benchmarks')
    return parser.parse_args()


def load_model(model_path, device=''):
    """Load YOLOv8 model with optimizations"""
    print(f'Loading model: {model_path}')
    model = YOLO(model_path)
    
    # Set device
    if device:
        model.to(device)
    elif torch.cuda.is_available():
        model.to('cuda:0')
        print('Using CUDA device')
    else:
        model.to('cpu')
        print('Using CPU')
    
    # Warm up model
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_input, verbose=False)
    print('Model loaded and warmed up')
    
    return model


def process_video(model, source, args):
    """Process video source (webcam or video file)"""
    # Open video source
    cap = cv2.VideoCapture(source if source != '0' else 0)
    if not cap.isOpened():
        raise ValueError(f'Failed to open video source: {source}')
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or args.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video: {width}x{height} @ {fps} FPS')
    
    # Setup output writer if saving
    writer = None
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'output_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f'Saving to: {output_path}')
    
    # FPS tracking
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 30
    
    print('Starting inference... Press ESC to quit')
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('End of video or failed to read frame')
                break
            
            # Run inference
            results = model.predict(
                frame,
                imgsz=tuple(args.imgsz),
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                classes=args.classes,
                verbose=False
            )
            
            # Visualize results
            annotated_frame = draw_boxes(
                frame,
                results[0],
                hide_labels=args.hide_labels,
                hide_conf=args.hide_conf,
                line_thickness=args.line_thickness
            )
            
            # Display
            if args.show:
                cv2.imshow('YOLOv8 Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    print('User interrupted')
                    break
            
            # Save
            if writer:
                writer.write(annotated_frame)
            
            # FPS benchmark
            frame_count += 1
            if args.benchmark and frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f'FPS: {current_fps:.2f} | Frame: {frame_count}')
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f'\nProcessed {frame_count} frames in {total_time:.2f}s')
        print(f'Average FPS: {avg_fps:.2f}')


def process_images(model, image_folder, args):
    """Process folder of images"""
    image_folder = Path(image_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = [p for p in image_folder.rglob('*') if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f'No images found in {image_folder}')
        return
    
    print(f'Found {len(image_paths)} images')
    
    # Setup output directory
    output_dir = None
    if args.save:
        output_dir = Path(args.output_dir) / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving to: {output_dir}')
    
    # Process each image
    start_time = time.time()
    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Failed to read: {img_path}')
            continue
        
        # Run inference
        results = model.predict(
            img,
            imgsz=tuple(args.imgsz),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            classes=args.classes,
            verbose=False
        )
        
        # Visualize
        annotated_img = draw_boxes(
            img,
            results[0],
            hide_labels=args.hide_labels,
            hide_conf=args.hide_conf,
            line_thickness=args.line_thickness
        )
        
        # Display
        if args.show:
            cv2.imshow('YOLOv8 Detection', annotated_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print('User interrupted')
                break
        
        # Save
        if output_dir:
            output_path = output_dir / f'{img_path.stem}_detected{img_path.suffix}'
            cv2.imwrite(str(output_path), annotated_img)
        
        if args.benchmark and idx % 10 == 0:
            print(f'Processed {idx}/{len(image_paths)} images')
    
    if args.show:
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f'\nProcessed {len(image_paths)} images in {total_time:.2f}s')
    print(f'Average time per image: {total_time/len(image_paths):.3f}s')


def main():
    args = parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Determine source type
    source_path = Path(args.source)
    
    if args.source in ['0', '1', '2'] or not source_path.exists():
        # Webcam or video stream
        process_video(model, args.source, args)
    elif source_path.is_file():
        # Video file
        process_video(model, str(source_path), args)
    elif source_path.is_dir():
        # Image folder
        process_images(model, source_path, args)
    else:
        raise ValueError(f'Invalid source: {args.source}')
    
    print('\nInference complete!')


if __name__ == '__main__':
    main()
