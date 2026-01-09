# Computer Vision Engineer Assignment

This repository contains solutions for the Computer Vision Engineer intern assignment, covering object detection, quality inspection, and vision-language model design.

## Tasks Overview

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Custom Object Detection from Scratch | Complete |
| Task 2 | Automated Quality Inspection (PCB Defects) | Complete |
| Task 3 | Custom VLM Design for Industrial Inspection | Complete |

---

## Task 1: Custom Object Detection with Model Training from Scratch

**File:** `Task_1.ipynb`

A complete object detection pipeline built from scratch using PyTorch, without pre-trained weights.

### Features
- Custom CNN-based detector architecture
- Bounding box regression with MSE loss
- Training on synthetic dataset with white rectangles
- Visualization of predictions with bounding boxes

### Architecture
```
SimpleDetector
├── Conv2d(1, 16) + ReLU + MaxPool
├── Conv2d(16, 32) + ReLU + MaxPool
├── Conv2d(32, 64) + ReLU + MaxPool
├── Flatten
├── Linear(64*8*8, 128) + ReLU
└── Linear(128, 4) + Sigmoid → [x, y, w, h]
```

### Results
- Training Loss convergence over 100 epochs
- Accurate bounding box predictions on test samples

---

## Task 2: Automated Quality Inspection System for Manufacturing

**File:** `Task_2.ipynb`

An automated visual inspection system for PCB (Printed Circuit Board) defect detection using YOLOv8.

### Defect Classes (6 types)
1. Missing Hole
2. Mouse Bite
3. Open Circuit
4. Short
5. Spur
6. Spurious Copper

### Features
- YOLOv8 model training on PCB defect dataset
- Defect localization with bounding boxes
- Confidence scores for each detection
- Severity assessment (Low/Medium/High)
- (x, y) pixel coordinates of defect centers

### Dataset
- Source: Roboflow PCB Defect Dataset
- Location: `PCB-Defect-1/`
- Split: Train/Test with YOLO format annotations

### Sample Output
```
--- Inspection Report ---
Defect: short | Confidence: 0.94 | Severity: High | Center: (256, 172)
Defect: mouse_bite | Confidence: 0.87 | Severity: Medium | Center: (431, 303)
Total defects found: 2
```

---

## Task 3: Custom VLM Design for Industrial Quality Inspection

**File:** `Task_3.ipynb`

A comprehensive design document for building a custom Vision-Language Model (VLM) for PCB inspection.

### Scenario
- Offline AI system for semiconductor manufacturer
- Natural language queries about defects
- <2 second inference requirement
- 50,000 PCB images with bounding boxes available

### Design Document Sections

#### (A) Model Selection
- **Recommended:** Qwen-VL (7B) with custom modifications
- Comparison with LLaVA, BLIP-2, and custom architectures
- Native bounding box support for precise localization

#### (B) Design Strategy
- Multi-scale vision encoder with PCB-specific attention
- Domain vocabulary expansion for defect types
- Spatial-aware cross-modal fusion mechanism
- Structured JSON output format

#### (C) Optimization for <2s Inference
- AWQ INT4 quantization (5GB VRAM, ~1.1s latency)
- vLLM/TensorRT-LLM inference stack
- Flash Attention 2 integration
- Optional knowledge distillation to 3B model

#### (D) Hallucination Mitigation
- Grounded training with bbox supervision
- Hard negative mining strategy
- Uncertainty quantification head
- YOLO verification cross-check

#### (E) Training Plan
- 4-stage curriculum: Vision alignment → Instruction tuning → Hallucination reduction → Quantization-aware
- QA pair generation (200K+ pairs from 50K images)
- Data augmentation with Albumentations

#### (F) Validation Methodology
- Counting accuracy (>95% target)
- Localization mAP@50 (>85% target)
- Hallucination rate (<5% target)
- Latency P95 (<2000ms target)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aryankd123/CV-Engineer-Assignment.git
cd CV-Engineer-Assignment

# Install dependencies
pip install torch torchvision
pip install ultralytics
pip install roboflow
pip install opencv-python matplotlib numpy pyyaml
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- OpenCV
- Matplotlib
- NumPy

## Hardware Used
- Apple Silicon (MPS) for training
- Compatible with CUDA GPUs

## Directory Structure

```
CV-Engineer-Assignment/
├── README.md
├── Task_1.ipynb          # Custom object detection from scratch
├── Task_2.ipynb          # PCB defect detection with YOLOv8
├── Task_3.ipynb          # VLM design document
└── PCB-Defect-1/         # PCB defect dataset
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

## Author

Aryan

## License

MIT License
