#!/usr/bin/env bash
# Detect-and-adapt setup for the lane-detection ADAS stack (Jetson / CUDA hosts).
# Flow: detect → ensure CUDA/TRT → system-site-packages venv → pip deps → verify →
#       inventory ONNX under models/ → build missing TensorRT engines
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REBUILD_ENGINES=0
for arg in "$@"; do
  case "$arg" in
    --rebuild-engines) REBUILD_ENGINES=1 ;;
    -h|--help)
      cat <<EOF
Usage: ./setup.sh [--rebuild-engines]

  --rebuild-engines   Rebuild TensorRT engines even if the .engine file exists
EOF
      exit 0
      ;;
  esac
done

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }
section() {
  echo ""
  echo "========================================================"
  echo "$*"
  echo "========================================================"
}

APT_OPTS=(-y -o Acquire::Http::Timeout=120 -o Acquire::Retries=10 --fix-missing)
NEED_SUDO=1
if [[ "$(id -u)" -eq 0 ]]; then
  NEED_SUDO=0
fi
run_root() {
  if [[ "$NEED_SUDO" -eq 1 ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Detect machine / CUDA / TensorRT
# ---------------------------------------------------------------------------
detect_env() {
  section "Detecting system"

  ARCH="$(uname -m)"
  IS_JETSON=0
  if [[ -f /etc/nv_tegra_release ]] || grep -qi tegra /proc/device-tree/model 2>/dev/null; then
    IS_JETSON=1
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. Install/load an NVIDIA driver first."
  fi

  DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | awk -F'CUDA Version: ' '/CUDA Version/{print $2}' | awk '{print $1; exit}')"
  [[ -n "$DRIVER_CUDA" ]] || die "Could not read max CUDA version from nvidia-smi."

  DRIVER_CUDA_MAJOR="${DRIVER_CUDA%%.*}"
  DRIVER_CUDA_MINOR="${DRIVER_CUDA#*.}"
  DRIVER_CUDA_MINOR="${DRIVER_CUDA_MINOR%%.*}"

  CUDA_HOME=""
  for cand in \
    "/usr/local/cuda-${DRIVER_CUDA}" \
    "/usr/local/cuda-${DRIVER_CUDA_MAJOR}.${DRIVER_CUDA_MINOR}" \
    "/usr/local/cuda" \
    "/usr/local/cuda-${DRIVER_CUDA_MAJOR}"; do
    if [[ -x "${cand}/bin/nvcc" ]]; then
      CUDA_HOME="$cand"
      break
    fi
  done

  NVCC_VER=""
  if [[ -n "$CUDA_HOME" ]]; then
    NVCC_VER="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | awk '/release/{print $5}' | tr -d ',' || true)"
  elif command -v nvcc >/dev/null 2>&1; then
    NVCC_VER="$(nvcc --version 2>/dev/null | awk '/release/{print $5}' | tr -d ',' || true)"
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi

  TRTEXEC="$(command -v trtexec 2>/dev/null || true)"
  if [[ -z "$TRTEXEC" ]]; then
    for p in /usr/src/tensorrt/bin/trtexec /usr/lib/aarch64-linux-gnu/bin/trtexec; do
      if [[ -x "$p" ]]; then
        TRTEXEC="$p"
        break
      fi
    done
  fi

  HAS_SYSTEM_TRT=0
  SYSTEM_TRT_VER=""
  if python3 -c "import tensorrt" >/dev/null 2>&1; then
    HAS_SYSTEM_TRT=1
    SYSTEM_TRT_VER="$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)"
  fi

  log "arch=${ARCH}  jetson=${IS_JETSON}"
  log "driver=${DRIVER_VER:-unknown}  driver_max_cuda=${DRIVER_CUDA}"
  log "cuda_home=${CUDA_HOME:-missing}  nvcc=${NVCC_VER:-missing}"
  log "trtexec=${TRTEXEC:-missing}"
  log "system_python_tensorrt=${SYSTEM_TRT_VER:-missing}"
}

# ---------------------------------------------------------------------------
# Ensure base apt packages (safe / common)
# ---------------------------------------------------------------------------
ensure_base_packages() {
  section "Ensuring base packages"
  run_root apt-get update
  run_root apt-get install "${APT_OPTS[@]}" \
    git \
    git-lfs \
    python3 \
    python3-venv \
    python3-opencv \
    libcurl4-openssl-dev \
    libsqlite3-dev \
    curl \
    ca-certificates
  git lfs install || true
}

# ---------------------------------------------------------------------------
# Ensure CUDA toolkit matching driver max CUDA (do not install newer)
# ---------------------------------------------------------------------------
ensure_cuda() {
  section "Ensuring CUDA toolkit (<= driver ${DRIVER_CUDA})"

  if [[ -n "$CUDA_HOME" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    log "CUDA toolkit already present at ${CUDA_HOME} (nvcc ${NVCC_VER})"
  else
    PKG="cuda-toolkit-${DRIVER_CUDA_MAJOR}-${DRIVER_CUDA_MINOR}"
    if apt-cache show "$PKG" >/dev/null 2>&1; then
      log "Installing ${PKG} to match driver CUDA ${DRIVER_CUDA}"
      run_root apt-get install "${APT_OPTS[@]}" "$PKG"
    else
      warn "Package ${PKG} not found; trying meta package cuda-toolkit"
      run_root apt-get install "${APT_OPTS[@]}" cuda-toolkit || \
        die "Could not install a CUDA toolkit matching driver CUDA ${DRIVER_CUDA}"
    fi

    CUDA_HOME=""
    for cand in \
      "/usr/local/cuda-${DRIVER_CUDA}" \
      "/usr/local/cuda-${DRIVER_CUDA_MAJOR}.${DRIVER_CUDA_MINOR}" \
      "/usr/local/cuda"; do
      if [[ -x "${cand}/bin/nvcc" ]]; then
        CUDA_HOME="$cand"
        break
      fi
    done
    [[ -n "$CUDA_HOME" ]] || die "CUDA toolkit install finished but nvcc was not found."
    NVCC_VER="$("$CUDA_HOME/bin/nvcc" --version | awk '/release/{print $5}' | tr -d ',')"
  fi

  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export CUDA_ROOT="${CUDA_HOME}"
  export CUDA_HOME
  log "Using CUDA_HOME=${CUDA_HOME}"
}

# ---------------------------------------------------------------------------
# Ensure TensorRT (prefer existing system packages; avoid conflicting metas)
# ---------------------------------------------------------------------------
ensure_tensorrt() {
  section "Ensuring TensorRT"

  if [[ -n "$TRTEXEC" && -x "$TRTEXEC" && "$HAS_SYSTEM_TRT" -eq 1 ]]; then
    log "Reusing system TensorRT ${SYSTEM_TRT_VER} and ${TRTEXEC}"
  else
    log "Installing TensorRT runtime packages (libnvinfer-bin, python3-libnvinfer)"
    # Prefer concrete packages. Avoid nvidia-tensorrt-dev when it conflicts
    # with an already-selected TensorRT/CUDA combo on the host.
    if ! run_root apt-get install "${APT_OPTS[@]}" libnvinfer-bin python3-libnvinfer; then
      warn "Primary TensorRT packages failed; retrying without forcing nvidia-tensorrt-dev"
      run_root apt-get install "${APT_OPTS[@]}" libnvinfer-bin python3-libnvinfer tensorrt || \
        die "TensorRT install failed. Fix apt TensorRT/CUDA conflicts, then re-run."
    fi
  fi

  TRTEXEC="$(command -v trtexec 2>/dev/null || true)"
  if [[ -z "$TRTEXEC" ]]; then
    for p in /usr/src/tensorrt/bin/trtexec /usr/lib/aarch64-linux-gnu/bin/trtexec; do
      if [[ -x "$p" ]]; then
        TRTEXEC="$p"
        break
      fi
    done
  fi
  [[ -n "$TRTEXEC" && -x "$TRTEXEC" ]] || die "trtexec not found after TensorRT setup."

  if python3 -c "import tensorrt" >/dev/null 2>&1; then
    HAS_SYSTEM_TRT=1
    SYSTEM_TRT_VER="$(python3 -c 'import tensorrt; print(tensorrt.__version__)')"
  else
    die "python3 cannot import system tensorrt (python3-libnvinfer)."
  fi

  export PATH="$(dirname "$TRTEXEC"):${PATH}"
  log "trtexec=${TRTEXEC}"
  log "system_tensorrt=${SYSTEM_TRT_VER}"
}

# ---------------------------------------------------------------------------
# Git LFS model weights + ONNX inventory
# ---------------------------------------------------------------------------
# ADAS GUI engines (ONNX → TensorRT). Extra ONNX on disk that the app does not
# load (e.g. yolo26n-reid) is listed but not compiled.
LANE_ONNX="models/anchor3dlane_raw.onnx"
LANE_ENGINE="models/anchor3dlane_raw.engine"

fetch_lfs() {
  section "Fetching Git LFS model weights"
  if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
    git lfs install
    git lfs pull || warn "git lfs pull failed; ensure ONNX files exist under models/"
  else
    warn "git-lfs not available; skipping LFS pull"
  fi

  mkdir -p models
  [[ -f "$LANE_ONNX" ]] || die "Missing ${LANE_ONNX} (lane detector)"
}

model_status() {
  local onnx="$1"
  local engine="$2"
  local role="$3"
  local required="${4:-0}"
  if [[ -f "$onnx" ]]; then
    if [[ -f "$engine" && "$REBUILD_ENGINES" -eq 0 ]]; then
      log "HAVE  ${role}: ${onnx}  →  ${engine}"
    elif [[ -f "$engine" ]]; then
      log "REBUILD ${role}: ${onnx}  →  ${engine}"
    else
      log "BUILD ${role}: ${onnx}  →  ${engine} (engine missing)"
    fi
  else
    if [[ "$required" -eq 1 ]]; then
      die "Missing required ONNX for ${role}: ${onnx}"
    fi
    warn "SKIP  ${role}: missing ${onnx}"
  fi
}

inventory_models() {
  section "Checking ONNX models under models/"
  log "rebuild_engines=${REBUILD_ENGINES}"
  model_status "$LANE_ONNX" "$LANE_ENGINE" "lanes" 1
  model_status models/yolov8n.onnx models/yolov8n.engine "vehicles (YOLO)" 0
  model_status models/traffic_sign_yolo11n.onnx models/traffic_sign_yolo11n.engine "traffic signs" 0
  model_status models/us_lprnet_baseline18.onnx models/us_lprnet_baseline18.engine "LPRNet (US plates)" 0
  model_status models/midas_small.onnx models/monocular_depth.engine "depth (optional)" 0
  if [[ -f models/yolo26n-reid.onnx ]]; then
    log "EXTRA models/yolo26n-reid.onnx (not used by the ADAS app — not compiled)"
  fi
  if [[ -f models/lprnet_dict_us.txt ]]; then
    log "HAVE  LPRNet charset: models/lprnet_dict_us.txt"
  else
    warn "Missing models/lprnet_dict_us.txt (LPRNet decode needs it)"
  fi
}

# ---------------------------------------------------------------------------
# Python venv: use --system-site-packages so system TensorRT/OpenCV are visible
# ---------------------------------------------------------------------------
venv_uses_system_site() {
  local cfg="$ROOT/venv/pyvenv.cfg"
  [[ -f "$cfg" ]] || return 1
  grep -qiE '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "$cfg"
}

ensure_venv() {
  section "Setting up Python venv (system-site-packages)"

  local recreate=0
  if [[ ! -d venv ]]; then
    recreate=1
  elif ! venv_uses_system_site; then
    warn "Existing venv does not include system site-packages (needed for system TensorRT)."
    warn "Recreating venv (old venv backed up to venv.bak.$$)."
    mv venv "venv.bak.$$"
    recreate=1
  fi

  if [[ "$recreate" -eq 1 ]]; then
    python3 -m venv --system-site-packages venv
  else
    log "Reusing existing system-site-packages venv"
  fi

  ./venv/bin/python -m pip install --upgrade pip setuptools wheel
}

# ---------------------------------------------------------------------------
# Install requirements; never keep pip TensorRT on Jetson / system-TRT hosts
# ---------------------------------------------------------------------------
install_python_deps() {
  section "Installing Python requirements"

  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export CUDA_ROOT="${CUDA_HOME}"

  ./venv/bin/pip install -r requirements.txt

  # Remove pip TensorRT wheels that shadow system 10.x bindings.
  ./venv/bin/pip uninstall -y tensorrt tensorrt-cu12 tensorrt-cu11 \
    tensorrt_libs tensorrt_bindings 2>/dev/null || true

  local pyver
  pyver="$(./venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  rm -rf \
    "./venv/lib/python${pyver}/site-packages/tensorrt" \
    "./venv/lib/python${pyver}/site-packages"/tensorrt-*.dist-info \
    "./venv/lib/python${pyver}/site-packages"/tensorrt_libs* \
    "./venv/lib/python${pyver}/site-packages"/tensorrt_bindings* \
    2>/dev/null || true

  log "Python deps installed; pip TensorRT removed so system TensorRT is used"
}

# ---------------------------------------------------------------------------
# Verify runtime imports before building engines
# ---------------------------------------------------------------------------
verify_runtime() {
  section "Verifying runtime"

  ./venv/bin/python - <<'PY'
import importlib
import sys

mods = ["tensorrt", "cv2", "numpy", "PySide6", "PIL", "ultralytics"]
missing = []
for name in mods:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"OK  {name:14} {ver}")
    except Exception as e:
        missing.append((name, str(e)))
        print(f"MISS {name:14} {e}")

if missing:
    sys.exit(1)

import tensorrt as trt
print("using_tensorrt", trt.__version__)
PY

  [[ -x "$TRTEXEC" ]] || die "trtexec missing at verify step"
  log "Runtime OK"
}

# ---------------------------------------------------------------------------
# Build TensorRT engines on THIS machine (never copy .engine across hosts)
# ---------------------------------------------------------------------------
build_engine() {
  local onnx="$1"
  local engine="$2"
  shift 2 || true
  if [[ ! -f "$onnx" ]]; then
    warn "Skip engine (missing ONNX): $onnx"
    return 0
  fi
  if [[ -f "$engine" && "$REBUILD_ENGINES" -eq 0 ]]; then
    log "Skip existing engine: $engine"
    return 0
  fi
  if [[ -f "$engine" && "$REBUILD_ENGINES" -eq 1 ]]; then
    log "Rebuilding $engine from $onnx"
  else
    log "Building $engine from $onnx"
  fi
  "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --fp16 "$@"
}

build_yolo_engine() {
  if [[ -f models/yolov8n.engine && "$REBUILD_ENGINES" -eq 0 ]]; then
    log "Skip existing engine: models/yolov8n.engine"
    return 0
  fi
  if [[ -f models/yolov8n.onnx ]]; then
    build_engine models/yolov8n.onnx models/yolov8n.engine
    return 0
  fi
  if [[ -f scripts/export_yolo_orin.py ]]; then
    log "No yolov8n.onnx — exporting and compiling via scripts/export_yolo_orin.py"
    ./venv/bin/python scripts/export_yolo_orin.py --imgsz 640
    return 0
  fi
  warn "Skip vehicles engine (no models/yolov8n.onnx and no export script)"
}

build_engines() {
  section "Building TensorRT engines (FP16) from ONNX"

  mkdir -p models
  inventory_models

  build_engine "$LANE_ONNX" "$LANE_ENGINE"
  build_yolo_engine
  build_engine models/traffic_sign_yolo11n.onnx models/traffic_sign_yolo11n.engine
  build_engine models/us_lprnet_baseline18.onnx models/us_lprnet_baseline18.engine \
    --minShapes=image_input:1x3x48x96 \
    --optShapes=image_input:1x3x48x96 \
    --maxShapes=image_input:1x3x48x96
  build_engine models/midas_small.onnx models/monocular_depth.engine
}

print_next_steps() {
  section "Setup complete"
  cat <<EOF
Detected:
  arch=${ARCH}  jetson=${IS_JETSON}
  driver_cuda=${DRIVER_CUDA}  cuda_home=${CUDA_HOME}
  tensorrt=${SYSTEM_TRT_VER}  trtexec=${TRTEXEC}

Run:
  source venv/bin/activate
  python scripts/run_pyside6_app.py

Other:
  python scripts/run_pyside6_app.py --video testing_new_videos/GRMN6695_540_nohud.mp4
  python scripts/run_pyside6_app.py --test-mode

Rebuild engines later (if needed):
  ./setup.sh --rebuild-engines
EOF
}

# ---------------------------------------------------------------------------
main() {
  section "Lane detection ADAS detect-and-adapt setup"
  if [[ "$REBUILD_ENGINES" -eq 1 ]]; then
    detect_env
    [[ -n "$TRTEXEC" && -x "$TRTEXEC" ]] || die "trtexec not found. Run ./setup.sh once without --rebuild-engines."
    fetch_lfs
    build_engines
    print_next_steps
    return
  fi
  detect_env
  ensure_base_packages
  ensure_cuda
  ensure_tensorrt
  fetch_lfs
  ensure_venv
  install_python_deps
  verify_runtime
  build_engines
  print_next_steps
}

main "$@"
