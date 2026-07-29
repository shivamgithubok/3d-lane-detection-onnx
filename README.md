# 🛣️ Anchor3DLane Jetson Inference & TensorRT GPU Acceleration + YOLO ByteTrack CIPO Pipeline

This repository provides real-time 3D lane detection and **CIPO (Closest In-Path Object)** collision alert utilities optimized for **NVIDIA Jetson Orin** edge GPUs. It features high-speed TensorRT `.engine` GPU acceleration, 3D-to-2D perspective projection onto front-view camera frames, real-world Bird's Eye View (BEV) mapping in meters, **YOLO ByteTrack multi-object tracking (Cars & Trucks)**, and real-time CIPO alert telemetry.

---

## 🎬 Live Pipeline Demo Video

[![CIPO Pipeline Demo Video](output/example_3_cipo_bytetrack_fixed.mp4)](file:///home/elevatics/Projects/3d-lane-detection-onnx/output/example_3_cipo_bytetrack_fixed.mp4)

> **Demo Output Video:** [output/example_3_cipo_bytetrack_fixed.mp4](file:///home/elevatics/Projects/3d-lane-detection-onnx/output/example_3_cipo_bytetrack_fixed.mp4)
> Features real-time 3-panel split view: **Front View** (3D lane lines + ByteTrack bounding boxes + CIPO alert header), **Bird's Eye View (BEV)** (real-world 3D lane grid & vehicle positions in meters), and **Telemetry HUD** (live FPS & CIPO range).

---

## 📊 Performance Benchmark Results (NVIDIA Jetson Orin)

| Model Execution Mode | Platform / Engine | Latency (ms) | Inference Speed | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **CPU ONNX Inference** | ARM Cortex CPU | ~1,665 ms | ~0.6 FPS | Baseline |
| **3D Lane Model Only (TensorRT FP16)** | **NVIDIA Orin GPU** | **~9.6 ms** | **103.7 FPS (Peak)** / **74.7 FPS (Sustained)** | **~125x Faster** 🚀 |
| **End-to-End Real-Time Video (3D Lane Only)** | **NVIDIA Orin GPU** | **~28.5 ms** | **~35.2 FPS (Live Stream)** | Real-Time Real-World |
| **3D Lane + YOLO ByteTrack CIPO Pipeline** | **NVIDIA Orin GPU** | **~30.1 ms** | **~33.1 FPS (Live Stream)** | Full CIPO ADAS System |

---

## 📁 Repository Directory Structure

```text
3d-lane-detection-onnx/
├── data/
│   └── images/                     # Input images and sample MP4 videos
│       ├── example.jpg
│       ├── example_1.mp4
│       ├── example_2.mp4
│       └── example_3.mp4
├── models/
│   ├── anchor3dlane_raw.onnx        # ONNX Model weights file (Git LFS)
│   ├── anchor3dlane_raw.engine      # Compiled TensorRT FP16 GPU Engine
│   └── yolov8n.pt                   # YOLO PyTorch model weights for vehicle tracking
├── output/                         # Generated annotated images & video outputs
│   ├── example_3_cipo_bytetrack_fixed.mp4 # CIPO Pipeline output video
│   ├── example_2_annotated.mp4
│   └── tensorrt_annotated.jpg
├── scripts/
│   ├── infer_cipo_pipeline.py      # Real-Time 3D Lane + YOLO ByteTrack CIPO Pipeline
│   ├── infer_video_tensorrt.py     # Real-time TensorRT video stream + live FPS & BEV map
│   ├── infer_tensorrt.py           # TensorRT GPU inference on single image
│   ├── benchmark_fps.py            # Latency and FPS benchmarking script
│   ├── tune_calibration.py         # Camera extrinsic/intrinsic calibration tuner
│   └── extract_frame.py            # Video frame extraction utility
├── src/
│   ├── inference/
│   │   ├── cipo_tracker.py         # 3D ground projection & lane ROI in-path filtering
│   │   ├── object_detector.py      # YOLO ByteTrack detector (Car & Truck tracking)
│   │   └── postprocess.py          # Softmax, NMS, and 3D-to-2D projection decoding
│   └── utils/
│       ├── calibration.py          # Camera pitch & height 3x4 projection matrix P
│       ├── split_visualization.py  # 3-Panel Split Window UI (Front View + BEV + HUD)
│       └── visualization.py        # Top-down Bird's Eye View (BEV) rendering (meters)
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation & usage guide
```

---

## 🚀 Quick Start (Automatic Setup & Run)

Run the automated setup script to install system dependencies, fetch model weights via Git LFS, configure Python virtual environment, and run inference:

```bash
# 1. Run 1-step automatic setup
chmod +x setup.sh
./setup.sh

# 2. Run 3D Lane + YOLO ByteTrack CIPO Pipeline (New)
./venv/bin/python3 scripts/infer_cipo_pipeline.py --video data/images/example_3.mp4 --output output/example_3_cipo_bytetrack_fixed.mp4

# 3. Run 3D Lane + YOLO ByteTrack CIPO Pipeline (Headless Mode)
./venv/bin/python3 scripts/infer_cipo_pipeline.py --video data/images/example_3.mp4 --output output/example_3_cipo_bytetrack_fixed.mp4 --no-gui

# 4. Run Standalone 3D Lane Video Inference
./venv/bin/python3 scripts/infer_video_tensorrt.py data/images/example_2.mp4

# 5. Run Single Image TensorRT GPU Inference
./venv/bin/python3 scripts/infer_tensorrt.py
```

---

## 🛠️ Manual Step-by-Step Setup

### 1. System Dependencies Installation

Install the required CUDA development libraries, TensorRT, and OpenCV GUI packages:

```bash
# Update APT and install CUDA libraries, TensorRT binaries, and OpenCV GUI
sudo apt-get update && sudo apt-get install -y \
    cuda-toolkit \
    nvidia-tensorrt-dev \
    libnvinfer-bin \
    python3-libnvinfer \
    python3-opencv \
    libcurl4-openssl-dev \
    libsqlite3-dev
```

### 2. Python Virtual Environment Setup

Create and set up the Python virtual environment:

```bash
# Create virtual environment
python3 -m venv --without-pip venv

# Install pip and required Python packages
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./venv/bin/python3 get-pip.py && rm get-pip.py

# Install Python requirements & dependencies
./venv/bin/pip install "numpy<2" opencv-python pycuda onnxruntime ultralytics torch torchvision

# Link system TensorRT and OpenCV modules to venv
cp -r /usr/lib/python3.12/dist-packages/tensorrt* ./venv/lib/python3.12/site-packages/
cp -r /usr/lib/python3/dist-packages/cv2* ./venv/lib/python3.12/site-packages/
```

### 3. Fetching Model Weights (Git LFS)

Ensure Git LFS is installed and pull the actual 52 MB `.onnx` binary weights file:

```bash
git lfs install
git lfs pull
```

---

## ⚡ Converting ONNX Model to TensorRT Engine (`.engine`)

Compile the `.onnx` model into a hardware-optimized TensorRT FP16 engine directly for your Jetson GPU using `trtexec`:

```bash
trtexec --onnx=models/anchor3dlane_raw.onnx \
        --saveEngine=models/anchor3dlane_raw.engine \
        --fp16
```

---

## 🚀 Running Inference & Benchmarking

### 1. 3D Lane + YOLO ByteTrack CIPO Pipeline
Run the full CIPO detection and tracking pipeline:

```bash
./venv/bin/python3 scripts/infer_cipo_pipeline.py --video data/images/example_3.mp4 --output output/example_3_cipo_bytetrack_fixed.mp4
```
> **Features:**
> - **ByteTrack Multi-Object Tracking:** Tracks Cars & Trucks with persistent tracking IDs (`Car #1`, `Truck #2`).
> - **3D Lane ROI Filtering:** Projects 2D vehicle boxes to 3D ground space $(X, Y)$ meters and tests if they lie inside the driving lane corridor.
> - **CIPO Distance Telemetry:** Identifies the Closest In-Path Object and triggers proportional risk alerts (`DANGER` $<15\text{m}$, `WARNING` $15-30\text{m}$, `SAFE` $>30\text{m}$).

### 2. Standalone 3D Lane TensorRT GPU Inference
Run real-time 3D lane inference with live FPS overlay and Bird's Eye View (BEV) visualization:

```bash
./venv/bin/python3 scripts/infer_video_tensorrt.py data/images/example_2.mp4
```

### 3. Benchmark Core GPU Engine FPS via `trtexec`
Measure pure GPU throughput over 100+ iterations:

```bash
trtexec --loadEngine=models/anchor3dlane_raw.engine \
        --shapes=img:1x3x360x480,mask:1x1x360x480 \
        --iterations=100
```

---

## 📚 Acknowledgements & References

- **Model Training & Architecture:** Based on [Anchor3DLane](https://github.com/tusen-ai/anchor3dlane), a 3D anchor-based lane detection framework predicting 3D lane proposals along longitudinal Y-anchor steps.
- **Dataset:** Trained using the [OpenLane dataset](https://github.com/OpenDriveLab/OpenLane), constructed from Waymo Open Dataset with 3D lane annotations and camera calibration parameters.
