#!/usr/bin/env bash
set -e

echo "========================================================"
echo "🛣️ Anchor3DLane Jetson TensorRT Auto-Setup Script"
echo "========================================================"

# 1. Install System Dependencies via APT
echo ""
echo "[1/5] Checking and installing system packages (CUDA, TensorRT, OpenCV)..."
sudo apt-get update && sudo apt-get install -y -o Acquire::Http::Timeout=120 -o Acquire::Retries=10 --fix-missing \
    cuda-toolkit \
    nvidia-tensorrt-dev \
    libnvinfer-bin \
    python3-libnvinfer \
    python3-opencv \
    git-lfs \
    libcurl4-openssl-dev \
    libsqlite3-dev

# 2. Pull Git LFS Model Weights
echo ""
echo "[2/5] Fetching ONNX model weights via Git LFS..."
git lfs install
git lfs pull

# 3. Setup Python Virtual Environment
echo ""
echo "[3/5] Setting up Python virtual environment (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv --without-pip venv
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    ./venv/bin/python3 get-pip.py && rm -f get-pip.py
fi

# 4. Install Python dependencies
echo ""
echo "[4/5] Installing Python dependencies from requirements.txt..."
export PATH=/usr/local/cuda-13.2/bin:$PATH
export CUDA_ROOT=/usr/local/cuda-13.2
./venv/bin/pip install -r requirements.txt

# Link system TensorRT and OpenCV modules into venv
cp -r /usr/lib/python3.12/dist-packages/tensorrt* ./venv/lib/python3.12/site-packages/ 2>/dev/null || true
cp -r /usr/lib/python3/dist-packages/cv2* ./venv/lib/python3.12/site-packages/ 2>/dev/null || true

# 5. Compile TensorRT GPU Engine
echo ""
echo "[5/5] Compiling ONNX model into TensorRT FP16 GPU Engine..."
if [ ! -f "models/anchor3dlane_raw.engine" ]; then
    trtexec --onnx=models/anchor3dlane_raw.onnx \
            --saveEngine=models/anchor3dlane_raw.engine \
            --fp16
else
    echo "TensorRT engine already exists at models/anchor3dlane_raw.engine"
fi

echo ""
echo "========================================================"
echo "Setup Complete! Run any of these commands to test:"
echo "========================================================"
echo "  source venv/bin/activate"
echo "  1. PySide6 ADAS UI:   python scripts/run_pyside6_app.py --video data/images/example_3.mp4"
echo "  2. Image GPU Test:    python scripts/infer_tensorrt.py"
echo "  3. Video GPU Stream:  python scripts/infer_video_tensorrt.py data/images/example_2.mp4"
echo "  4. Engine Benchmark:  trtexec --loadEngine=models/anchor3dlane_raw.engine --shapes=img:1x3x360x480,mask:1x1x360x480 --iterations=100"
echo "========================================================"
