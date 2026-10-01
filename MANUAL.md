# Manual

Companion to [README.md](README.md). The README is the intro and the commands you run day to day. This file is what you open when those commands fail, plus the reference behind them.

**Help** below is only the failure path: symptom, then the shortest fix, then a link to the full section. Do not start in the reference sections unless Help sends you there.

---

## Help

### Setup fails

`./setup.sh` errors, `source venv/bin/activate` has no venv, or Python cannot import TensorRT / OpenCV / PySide6.

1. Re-run setup. It detects the Jetson CUDA/TensorRT stack and will not force a toolkit newer than the driver:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   source venv/bin/activate
   ```

2. If packages are present and you only need a clean engine rebuild (skips APT and the venv):

   ```bash
   ./setup.sh --rebuild-engines
   ```

3. Manual pip install, only if you are not using `setup.sh`:

   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

The venv is created with `--system-site-packages`. `setup.sh` then removes pip `tensorrt` wheels so they do not shadow system `python3-libnvinfer`. Details: [Dependencies](#dependencies) and [What `setup.sh` does](#what-setupsh-does).

### Engines missing or won't build

The GUI starts, but lanes, vehicles, signs, or plates are absent. Engines must be built **on this Orin**. Do not copy `.engine` files from another machine.

```bash
./setup.sh --rebuild-engines
```

Expected files: `models/anchor3dlane_raw.engine`, `models/yolov8n.engine`, `models/traffic_sign_yolo11n.engine`, `models/us_lprnet_baseline18.engine`, and optional `models/monocular_depth.engine`.

If the YOLO / sign / LPR ONNX is missing, export on-device, then rebuild:

```bash
source venv/bin/activate
python scripts/export_yolo_orin.py --imgsz 640
python scripts/export_traffic_sign_orin.py
python scripts/export_lprnet_orin.py
./setup.sh --rebuild-engines
```

Full `trtexec` commands and the model table: [TensorRT engine build](#tensorrt-engine-build).

### Lanes or BEV look wrong

Frames are stretch-resized to **480×360** after a **20% sky crop**. Leave `SKY_CROP_FRAC` at `0.20` in `src/utils/camera_transform.py`. Dropping the crop toward 10% collapses recall on the Garmin clips; past ~25% the model's 3D lane width starts to collapse.

`scripts/infer_video_tensorrt.py` is a full-frame resize with **no** sky crop. It is not the GUI path. Use `scripts/run_pyside6_app.py` or `scripts/infer_cipo_pipeline.py` when you need the deployment camera.

Check a clip through the real QML widget (build a detection cache once):

```bash
python scripts/debug/bev_poc_tune.py --cache
python scripts/debug/bev_qt3d_verify.py --metrics --video GRMN6695_540_nohud.mp4
python scripts/debug/crop_sweep.py
python scripts/debug/width_bias_probe.py
```

Why the crop, pose filter, and dash scroll behave this way: [BEV and camera notes](#bev-and-camera-notes).

---

## Dependencies

### Python (`requirements.txt`)

Installed automatically by `setup.sh` via `pip install -r requirements.txt`:

| Package | Purpose |
| :--- | :--- |
| `numpy<2.0.0` | Arrays / math |
| `opencv-python-headless>=4.6.0` | Video/image I/O & overlays |
| `pycuda>=2026.1` | CUDA bindings for TensorRT runtime |
| `onnxruntime>=1.18.0` | ONNX fallback / tooling |
| `protobuf>=4.25.0` | ONNX / model protobuf support |
| `flatbuffers>=23.5.26` | Runtime serialization |
| `PySide6>=6.6.0` | ADAS GUI (front cam + BEV) |
| `ultralytics>=8.0.0` | YOLO export / ByteTrack |

Manual install (if not using `setup.sh`):

```bash
source venv/bin/activate
pip install -r requirements.txt
```

> **Jetson note:** `setup.sh` creates the venv with `--system-site-packages` and removes pip `tensorrt` wheels so the system `python3-libnvinfer` is used. YOLO / depth rebuilds may need `ultralytics` (already in `requirements.txt`) and a Jetson-compatible `torch` if you rebuild detectors from `.pt` weights.

### System packages (via `setup.sh`)

`setup.sh` installs (through `apt`):

- `cuda-toolkit`
- `libnvinfer-bin`, `python3-libnvinfer`
- `python3-opencv`
- `git-lfs`
- `libcurl4-openssl-dev`, `libsqlite3-dev`

---

## What `setup.sh` does

Detect-and-adapt setup (safe across Jetsons with different CUDA/TensorRT stacks):

1. Detects GPU driver max CUDA, toolkit path, `trtexec`, and system Python TensorRT
2. Installs base APT packages (OpenCV, Git LFS, build deps)
3. Ensures a CUDA toolkit **≤ driver max CUDA** (does not force a newer toolkit)
4. Reuses system TensorRT when present; otherwise installs `libnvinfer-bin` + `python3-libnvinfer` (avoids conflicting `nvidia-tensorrt-dev` when possible)
5. Creates `venv` with `--system-site-packages` so system TensorRT/OpenCV are visible
6. Installs `requirements.txt`, then removes any pip TensorRT wheels that would shadow system TRT
7. Verifies imports, inventories ONNX under `models/`, then builds missing TensorRT engines (lanes, YOLO vehicles, traffic signs, LPRNet, optional MiDaS depth)

```bash
chmod +x setup.sh
./setup.sh

# Rebuild .engine files from existing ONNX (does not skip present engines)
./setup.sh --rebuild-engines
```

---

## TensorRT engine build

`setup.sh` compiles each ONNX that is present under `models/`. Engines must be built **on this Orin** (do not copy `.engine` files between machines).

```bash
# Lane detector
trtexec --onnx=models/anchor3dlane_raw.onnx \
        --saveEngine=models/anchor3dlane_raw.engine \
        --fp16

# Vehicles (YOLOv8n)
trtexec --onnx=models/yolov8n.onnx \
        --saveEngine=models/yolov8n.engine \
        --fp16

# Traffic signs (YOLOv11n)
trtexec --onnx=models/traffic_sign_yolo11n.onnx \
        --saveEngine=models/traffic_sign_yolo11n.engine \
        --fp16

# US LPRNet (speed-limit plates) — fixed 48×96 input
trtexec --onnx=models/us_lprnet_baseline18.onnx \
        --saveEngine=models/us_lprnet_baseline18.engine \
        --fp16 \
        --minShapes=image_input:1x3x48x96 \
        --optShapes=image_input:1x3x48x96 \
        --maxShapes=image_input:1x3x48x96

# Optional depth
trtexec --onnx=models/midas_small.onnx \
        --saveEngine=models/monocular_depth.engine \
        --fp16
```

Lane benchmark:

```bash
trtexec --loadEngine=models/anchor3dlane_raw.engine \
        --shapes=img:1x3x360x480,mask:1x1x360x480 \
        --iterations=100
```

Expected models for the full PySide6 pipeline:

| File | Role |
| :--- | :--- |
| `models/anchor3dlane_raw.engine` | 3D lanes |
| `models/yolov8n.engine` | Vehicles (ByteTrack, imgsz=640) |
| `models/traffic_sign_yolo11n.engine` | Traffic signs / ISA |
| `models/us_lprnet_baseline18.engine` | US LPRNet digit OCR |
| `models/lprnet_dict_us.txt` | LPRNet charset |
| `models/monocular_depth.engine` | Depth (optional) |

If YOLO ONNX is missing, export then compile:

```bash
source venv/bin/activate
python scripts/export_yolo_orin.py --imgsz 640
python scripts/export_traffic_sign_orin.py
python scripts/export_lprnet_orin.py
```

---

## BEV and camera notes

| Piece | What it does |
| :--- | :--- |
| `src/tracking/lane_frame.py` | Pins the lane, measures ego offset/yaw from the corridor cubic, scrolls dashes from speed |
| `src/ui/bev_quick3d.py` + `src/ui/qml/BevScene.qml` | Qt Quick 3D BEV; ego `egoX` / `egoYawDeg`; no static ±5.35 m fallback edges |
| `src/utils/camera_transform.py` | Default `SKY_CROP_FRAC = 0.20` then resize to 480×360 |
| `src/utils/ego_speed.py` | HUD OCR speed for dash scroll and EKF coast |

Do **not** nudge the sky crop down toward 10% without re-running the sweep: that band collapses recall on the Garmin clips. Past ~25% the model's 3D lane width starts to collapse.

Verify a clip through the real QML widget (needs a detection cache first):

```bash
python scripts/debug/bev_poc_tune.py --cache          # once
python scripts/debug/bev_qt3d_verify.py --metrics --video GRMN6695_540_nohud.mp4
python scripts/debug/crop_sweep.py                    # sky-crop vs corridor quality
python scripts/debug/width_bias_probe.py              # raw ego-pair width vs 12 ft truth
```

---

## Repository layout

```text
3d-lane-detection-onnx/
├── data/                 # Example videos, 3D assets
├── testing_new_videos/   # Garmin dashcam targets (GRMN6694 / 6695 / 6700)
├── models/               # ONNX / TensorRT engines
├── scripts/
│   ├── run_pyside6_app.py          # Main GUI launcher
│   ├── infer_cipo_pipeline.py      # Uses CameraTransform (sky crop)
│   ├── export_yolo_orin.py
│   ├── infer_video_tensorrt.py
│   └── debug/                      # BEV tune / QML verify / crop & width probes
├── src/
│   ├── ui/               # PySide6 + Qt Quick 3D BEV
│   ├── inference/        # TRT / CIPO / YOLO / signs / LPRNet / LDW-FCW
│   ├── tracking/         # RoadStateEstimator + LaneFrameModel
│   └── utils/            # CameraTransform, ego speed, calibration
├── requirements.txt
├── setup.sh
├── README.md             # Intro and the commands you run
└── MANUAL.md             # Help, then reference
```

---

## Merge this branch into `main`

Current feature branch example: `Asphalt_view`.

### Option A — merge locally, then push `main`

```bash
# On your feature branch: commit & push first if needed
git checkout Asphalt_view
git status
git push -u origin Asphalt_view

# Merge into main
git checkout main
git pull origin main
git merge Asphalt_view

# Resolve conflicts if any, then:
git push origin main
```

### Option B — GitHub Pull Request (recommended)

```bash
git checkout Asphalt_view
git push -u origin Asphalt_view

gh pr create --base main --head Asphalt_view \
  --title "Asphalt BEV UI + CIPO dashboard updates" \
  --body "Cinematic BEV road, Anchor3D lane toggle, calibration defaults, FPS fixes."

# After review:
gh pr merge --merge
```

### After merge — sync feature branch (optional)

```bash
git checkout Asphalt_view
git merge main
git push origin Asphalt_view
```
