# 🛣️ Anchor3DLane Jetson Inference & TensorRT GPU Acceleration

This repository provides real-time 3D lane detection inference utilities optimized for **NVIDIA Jetson Orin** edge GPUs. It features high-speed TensorRT `.engine` GPU acceleration, 3D-to-2D perspective projection onto front-view camera frames, real-world Bird's Eye View (BEV) mapping in meters, and live FPS benchmarking.

---

## 📊 Performance Benchmark Results (NVIDIA Jetson Orin)

| Model Execution Mode | Platform / Engine | Latency (ms) | Inference Speed | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **CPU ONNX Inference** | ARM Cortex CPU | ~1,665 ms | ~0.6 FPS | Baseline |
| **TensorRT GPU Engine (FP16)** | **NVIDIA Orin GPU** | **~9.6 ms** | **103.7 FPS (Peak)** / **74.7 FPS (Sustained)** | **~125x Faster** 🚀 |
| **End-to-End Real-Time Video** | **NVIDIA Orin GPU** | **~28.5 ms** | **~35.2 FPS (Live Stream)** | Real-Time Real-World |

---

## 📁 Repository Directory Structure

```text
3d-lane-detection-onnx/
├── data/
│   └── images/                     # Input images and sample MP4 videos
│       ├── example.jpg
│       ├── example_1.mp4
│       └── example_2.mp4
├── models/
│   ├── anchor3dlane_raw.onnx        # ONNX Model weights file (Git LFS)
│   └── anchor3dlane_raw.engine      # Compiled TensorRT FP16 GPU Engine
├── output/                         # Generated annotated images & video outputs
│   ├── tensorrt_annotated.jpg
│   └── example_2_annotated.mp4
├── scripts/
│   ├── infer_tensorrt.py           # TensorRT GPU inference on single image
│   ├── infer_video_tensorrt.py     # Real-time TensorRT video stream + live FPS & BEV map
│   ├── benchmark_fps.py            # Latency and FPS benchmarking script
│   ├── infer_image.py              # ONNX Runtime single image inference
│   ├── infer_video.py              # ONNX Runtime video stream inference
│   ├── tune_calibration.py         # Camera extrinsic/intrinsic calibration tuner
│   ├── extract_frame.py            # Video frame extraction utility
│   └── debug/
│       ├── debug_projections.py    # Pixel coordinate projection validator
│       └── debug_scores.py         # Raw confidence score analyzer
├── src/
│   ├── inference/
│   │   └── postprocess.py          # Softmax, NMS, and 3D-to-2D projection decoding
│   └── utils/
│       ├── calibration.py          # Camera pitch & height 3x4 projection matrix P
│       └── visualization.py        # Top-down Bird's Eye View (BEV) rendering (meters)
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation & usage guide
```

---

## 🚀 Quick Start (Automatic Setup & Run)

Run the automated setup script to automatically install system dependencies, fetch model weights, configure Python `venv`, and compile the TensorRT GPU engine:

```bash
# 1. Run 1-step automatic setup
chmod +x setup.sh
./setup.sh

# 2. Run GPU TensorRT inference on an image
./venv/bin/python3 scripts/infer_tensorrt.py

# 3. Run real-time GPU TensorRT inference on video with live FPS & BEV display
./venv/bin/python3 scripts/infer_video_tensorrt.py data/images/example_2.mp4
```

---

## 🛠️ Manual Step-by-Step Setup

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

# Install Python requirements
./venv/bin/pip install "numpy<2" opencv-python-headless pycuda onnxruntime

# Link system TensorRT and OpenCV modules to venv
cp -r /usr/lib/python3.12/dist-packages/tensorrt* ./venv/lib/python3.12/site-packages/
cp -r /usr/lib/python3/dist-packages/cv2* ./venv/lib/python3.12/site-packages/
```

### 3. Fetching the Model Weights (Git LFS)

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

### 1. Single Image TensorRT GPU Inference
Run high-speed GPU inference on a single static image:

```bash
./venv/bin/python3 scripts/infer_tensorrt.py
```
> **Output:** Reports GPU inference time (~9-14 ms) and saves the annotated image to `output/tensorrt_annotated.jpg`.

### 2. Real-Time Video Stream GPU Inference
Run real-time video stream inference with live FPS overlay and Bird's Eye View (BEV) visualization:

```bash
./venv/bin/python3 scripts/infer_video_tensorrt.py data/images/example_2.mp4
```
> **Features:** 
> - Displays **Front View** with green 3D lane overlays and on-screen **Live GPU FPS counter**.
> - Displays **Bird's Eye View (BEV)** map plotting lane lines in real-world meters ($0\text{m} - 100\text{m}$).
> - Saves annotated video output to `output/example_2_annotated.mp4`.

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
