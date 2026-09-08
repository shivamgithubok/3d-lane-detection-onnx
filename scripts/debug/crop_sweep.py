#!/usr/bin/env python3
"""Sweep sky-crop and score it on downstream BEV quality, not lane counts.

The lane engine was trained on OpenLane framing (horizon near mid-frame). Dashcam
clips put the horizon much lower, so the road is squashed into the bottom of the
resized 480x360 input. Cropping sky restores training-like framing and can
recover recall - but Anchor3DLane regresses 3D directly from image position, so
a crop also risks biasing depth. A P-matrix change cannot undo that, because the
bias lives in the network's own 3D head.

So recall alone is not enough. The decisive check is ego lane width: real highway
lanes are ~3.7 m, and that number is known without any ground truth. If a crop
lifts recall while width stays near 3.7 m, the 3D survived. If width drifts, the
crop bought detections by corrupting geometry and must be rejected.

  CONFIRMED%   frames with a measured (not coasted) ego corridor - higher better
  width        mean ego lane width in metres - must stay near 3.7
  w_err        |width - 3.7| - the 3D bias indicator, lower better
  jit40        lane jitter at 40 m through LaneFrameModel - lower better

Usage:
  python scripts/debug/crop_sweep.py                        # all GRMN targets
  python scripts/debug/crop_sweep.py --videos data/images/example_3.mp4
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np

from src.inference import lane_filter_config as cfg
from src.inference.lane_preprocess import prepare_lane_input
from src.inference.postprocess import postprocess_onnx_output
from src.tracking.lane_frame import LaneFrameModel
from src.tracking.road_state import RoadStateEstimator
from src.utils.camera_transform import CameraTransform
from src.utils.ego_speed import EgoSpeedLog

INPUT_H, INPUT_W = 360, 480
ENGINE_PATH = "models/anchor3dlane_raw.engine"
TARGETS = [
    "testing_new_videos/GRMN6694_540_nohud.mp4",
    "testing_new_videos/GRMN6695_540_nohud.mp4",
    "testing_new_videos/GRMN6700_540_nohud.mp4",
]
PROBE_Y = np.array([40.0])


def load_trt():
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    with open(ENGINE_PATH, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    h_reg = np.empty((1, 4431, 86), dtype=np.float32)
    h_anc = np.empty((1, 4431, 65), dtype=np.float32)
    d_img = cuda.mem_alloc(3 * INPUT_H * INPUT_W * 4)
    d_mask = cuda.mem_alloc(INPUT_H * INPUT_W * 4)
    d_reg = cuda.mem_alloc(h_reg.nbytes)
    d_anc = cuda.mem_alloc(h_anc.nbytes)
    stream = cuda.Stream()
    ctx.set_tensor_address("img", int(d_img))
    ctx.set_tensor_address("mask", int(d_mask))
    ctx.set_tensor_address("reg_proposals", int(d_reg))
    ctx.set_tensor_address("anchors", int(d_anc))

    def infer(img, mask):
        cuda.memcpy_htod_async(d_img, np.ascontiguousarray(img), stream)
        cuda.memcpy_htod_async(d_mask, np.ascontiguousarray(mask), stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
        cuda.memcpy_dtoh_async(h_anc, d_anc, stream)
        stream.synchronize()
        return h_reg, h_anc

    return infer


def raw_lane_width(left, right):
    """Ego pair width straight from the model's 3D, before any clamping.

    RoadStateEstimator clamps width toward the target, which would mask exactly
    the bias we are trying to detect, so measure the unclamped pair.
    """
    if left is None or right is None:
        return np.nan
    L, R = np.asarray(left, float), np.asarray(right, float)
    n = min(len(L), len(R))
    if n < 2:
        return np.nan
    near = (L[:n, 1] >= 5.0) & (L[:n, 1] <= 30.0)
    if near.sum() < 2:
        return np.nan
    return float(np.mean(np.abs(R[:n][near, 0] - L[:n][near, 0])))


def run(infer, frames, fps, speed_log, crop_px):
    est = RoadStateEstimator()
    lf = LaneFrameModel()
    dt = 1.0 / max(1e-3, fps)
    confirmed, widths, x40, n_lanes = 0, [], [], []

    for i, frame in enumerate(frames):
        tf = CameraTransform.for_frame(frame, (INPUT_W, INPUT_H), sky_crop_px=crop_px)
        img, mask, meta = prepare_lane_input(tf.apply(frame))
        reg, _ = infer(img, mask)
        raw, _ = postprocess_onnx_output(reg, conf_threshold=meta["conf"])
        n_lanes.append(len(raw))
        sp = speed_log.get_mps(i) if speed_log is not None else None
        st = est.update(raw, dt=dt, speed_mps=sp)
        if st.status == "CONFIRMED":
            confirmed += 1
            w = raw_lane_width(st.left_corridor_3d, st.right_corridor_3d)
            if np.isfinite(w):
                widths.append(w)
        lf.update(st.left_corridor_3d, st.right_corridor_3d, speed_mps=sp, dt=dt)
        x40.append(float(lf.lane_x(PROBE_Y)[0]) if lf.valid else np.nan)

    n = len(frames)
    d = np.abs(np.diff(np.asarray(x40, float)))
    d = d[np.isfinite(d)]
    w_mean = float(np.mean(widths)) if widths else np.nan
    return {
        "conf_pct": 100.0 * confirmed / max(1, n),
        "lanes": float(np.mean(n_lanes)),
        "width": w_mean,
        "w_err": abs(w_mean - cfg.EGO_LANE_WIDTH_TARGET_M) if widths else np.nan,
        "jit40": float(d.mean()) if len(d) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=TARGETS)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--crops", type=float, nargs="*",
                    default=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    args = ap.parse_args()

    infer = load_trt()
    for video in args.videos:
        if not os.path.exists(video):
            print(f"missing {video}")
            continue
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames, idx = [], 0
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                frames.append(f)
            idx += 1
        cap.release()
        if not frames:
            continue
        speed_log = EgoSpeedLog.auto_load(video)
        h = frames[0].shape[0]
        print(f"\n=== {os.path.basename(video)}  {frames[0].shape[1]}x{h}  "
              f"{len(frames)} frames  speed_log={'yes' if speed_log else 'no'} ===")
        print(f"{'crop':>12} {'CONFIRMED%':>11} {'lanes':>7} {'width':>7} "
              f"{'w_err':>7} {'jit40':>8}")
        print("-" * 56)
        for frac in args.crops:
            px = int(frac * h)
            m = run(infer, frames, fps, speed_log, px)
            print(f"{int(frac*100):3d}% {px:4d}px {m['conf_pct']:10.1f}% "
                  f"{m['lanes']:7.2f} {m['width']:7.2f} {m['w_err']:7.2f} {m['jit40']:8.4f}")
    print(f"\n  width target = {cfg.EGO_LANE_WIDTH_TARGET_M} m; large w_err means the "
          f"crop biased the model's 3D and must be rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
