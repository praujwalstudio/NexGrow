# NexGrow: Field-Ready Domain-Adaptive Cotton Diagnostic System

> Every cotton leaf, one precise diagnosis.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![SAM 2](https://img.shields.io/badge/SAM-2-green)
![License: Internal](https://img.shields.io/badge/License-Internal-red)

NexGrow is an intelligent, enterprise-grade agricultural diagnostic platform specifically engineered to bridge the **"lab-to-field gap"** for cotton crops. It automates leaf detection, background isolation, and pathological analysis using a hierarchical multi-stage framework to provide reliable, real-time disease management in real-world agricultural environments.

---

## Table of Contents

- [What It Does](#-what-it-does)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Pipeline Flow](#-pipeline-flow)
- [Technical Report](#-technical-report)

---

## What It Does

Traditional AI models fail in real-world farms due to **domain shift** — variations in lighting, weather, and complex backgrounds. NexGrow replaces basic classification with a deterministic hierarchical loop:

1. **Ingests Imagery**: Captures high-resolution RGB data from UAVs or mobile devices in dense cotton fields.
2. **Assesses Quality**: Automatically filters blurred or low-clarity images to ensure feature extraction integrity.
3. **Localizes Leaves (YOLO11)**: Detects and crops individual cotton leaves within complex backgrounds (soil, weeds, shadows).
4. **Isolates Symptoms (SAM)**: Uses the Segment Anything Model for zero-shot isolation, creating a background-free input to prevent learning irrelevant noise.
5. **Generalizes Domain**: Applies Domain Generalization (DG) layers to maintain accuracy across different cotton varieties, seasons, and camera hardware.
6. **Executes Multi-Task Prediction**: Simultaneously identifies the Disease Type, Severity Level (Low/Moderate/High), and Confidence Score.
7. **Explains Reasoning**: Generates Grad-CAM++ and LIME heatmaps highlighting specific lesions for expert validation and farmer trust.
8. **Generates Weather-Aware Advice**: Retrieves localized treatment protocols based on the diagnosis and real-time environmental data.
9. **Edge Deployment**: Utilizes INT8 Quantization to compress the model to ~1.46 MB for offline, sub-millisecond mobile inference.

---

## Architecture

### System Topology

NexGrow operates on a **Teacher-Student Knowledge Distillation** architecture:

- **Teacher Core**: A powerful ensemble of Swin Transformer and ConvNeXt backbones to capture complex pathological features.
- **Student Core**: A lightweight MobileNetV4 or ShuffleNetV2 that inherits the teacher's intelligence while optimized for edge devices.
- **Quantization Layer**: Post-training compression via ONNX for mobile-native performance.

---

## Quick Start

### Core Dependencies

- Python 3.11+
- PyTorch 2.0.1+
- Ultralytics YOLO11 (Detection)
- Segment Anything (SAM/SAM2)
- FastAPI & Uvicorn (Backend API Gateway)

### Local Setup & Usage

```bash
# Clone the repository
git clone https://github.com/praujwalstudio/NexGrow.git
cd NexGrow

# Install diagnostic core and backend dependencies
pip install -r requirements.txt

# Start the high-performance FastAPI Server
# Set environment variables (optional: OPENWEATHER_API_KEY)
$env:NEXGROW_MODEL="models/student_model/dg_checkpoint.pth"
uvicorn server.main:app --reload

# Run End-to-End Inference CLI
python scripts/diagnose.py --image data/raw/leaf.jpg --model-path models/student_model/dg_checkpoint.pth --weather

# Run ONNX Latency & Model Size Benchmark CLI
python scripts/benchmark.py --model-path models/student_model/dg_checkpoint.pth --quantize
```

---

## Project Structure

```
NexGrow/
├── data/
│   ├── raw/                  # UAV & Kaggle datasets
│   ├── processed/            # SAM-isolated background-free leaves
│   └── augmented/            # AugMix & environmental simulations
├── models/
│   ├── detection/            # YOLO11 weights for leaf localization
│   ├── teacher_ensemble/     # Swin + ConvNeXt weights
│   ├── student_model/        # Distilled MobileNetV4
│   └── quantized/            # INT8 TFLite/ONNX models
├── src/
│   ├── preprocessing/        # Quality assessment & SAM isolation scripts
│   ├── training/             # Distillation & Domain Generalization
│   ├── explainability/       # Grad-CAM++ & LIME implementations
│   └── utils/                # Domain Generalization (MixStyle) layers
├── scripts/                  # CLI entry points (train, prepare, diagnose, benchmark)
├── server/                   # FastAPI high-performance Backend API Gateway & Docker
├── tests/                    # Unit, smoke, and full-pipeline integration tests
└── deployments/              # Edge deployment planning & docs
```

---

## Pipeline Flow

The NexGrow pipeline follows a strict contract-driven sequence to ensure diagnostic reliability:

1. **Detection Phase**: YOLO11 bounding box extraction
2. **Isolation Phase**: SAM binary mask generation and background removal
3. **Optimization Phase**: ADASYN-based class balancing and AugMix augmentation
4. **Learning Phase**: Teacher-to-Student knowledge distillation
5. **Validation Phase**: Cross-dataset testing (Kaggle → PlantDoc → Private UAV data)

### Diagnostic & Management Schema

Captured data must map to the following explicit variables for expert verification:

| Variable | Description |
|---|---|
| `pathogen_id` | Standardized taxonomy for 15 cotton diseases (e.g., Bacterial Blight, CLCuD) |
| `severity_index` | Three-point scale (Low, Moderate, High) to determine treatment urgency |
| `lesion_maps` | Heatmap arrays highlighting morphological symptoms for pathologists |
| `treatment_protocol` | Weather-aware recommendations (e.g., Fungicide if Humidity > 80%) |

---

## License

Internal development asset. For precision agriculture research use only.
