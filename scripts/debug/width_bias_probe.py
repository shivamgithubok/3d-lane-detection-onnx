#!/usr/bin/env python3
"""Is the lane engine's 3D laterally biased, and if so is it a constant scale?

Reads the ego pair straight from raw proposals via find_ego_lanes. It must NOT
go through extract_ego_corridor_3d: that clamps the pair and insets it by
EGO_CORRIDOR_MARGIN_M per side, which alone turns a correct 3.70 m lane into a
3.22 m corridor. Measuring the corridor and comparing it to 3.70 therefore
manufactures a ~0.5 m "bias" that is really a rendering inset.

The GRMN clips are US highways (CA-237, Milpitas), where lane width is
standardised at 12 ft = 3.66 m, so ground truth is known without labels.

Width is reported per depth band, because the shape of the error decides the fix:
  flat offset across bands  -> constant scale, correctable in calibration
  growing with distance     -> depth-dependent, needs fine-tuning

Usage:
  python scripts/debug/width_bias_probe.py
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np

from src.inference.lane_preprocess import prepare_lane_input
from src.inference.postprocess import postprocess_onnx_output
from src.utils.camera_transform import CameraTransform
from src.utils.drivable_area import find_ego_lanes, parse_lane_components

INPUT_H, INPUT_W = 360, 480
ENGINE_PATH = "models/anchor3dlane_raw.engine"
TRUE_LANE_M = 3.66  # US 12 ft
BANDS = [(3.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 45.0), (45.0, 65.0)]
TARGETS = [
    "testing_new_videos/GRMN6694_540_nohud.mp4",
    "testing_new_videos/GRMN6695_540_nohud.mp4",
    "testing_new_videos/GRMN6700_540_nohud.mp4",
]


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


def pair_widths(proposals):
    """Per-anchor (y, width) for the raw ego pair, unclamped and un-inset."""
    left, right = find_ego_lanes(proposals, 20)
    if left is None or right is None:
        return []
    xl, yl, _zl, vl = parse_lane_components(left, 20)
    xr, yr, _zr, vr = parse_lane_components(right, 20)
    if xl is None or xr is None:
        return []
    out = []
    n = min(len(xl), len(xr))
    for i in range(n):
        if not (vl[i] and vr[i]):
            continue
        y = float(yl[i])
        w = abs(float(xr[i]) - float(xl[i]))
        if 0.5 < w < 8.0:
            out.append((y, w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=TARGETS)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    infer = load_trt()
    pooled = {b: [] for b in BANDS}

    for video in args.videos:
        if not os.path.exists(video):
            print(f"missing {video}")
            continue
        cap = cv2.VideoCapture(video)
        samples, idx, taken = [], 0, 0
        while taken < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                samples.append(f)
                taken += 1
            idx += 1
        cap.release()
        if not samples:
            continue

        per_band = {b: [] for b in BANDS}
        for f in samples:
            tf = CameraTransform.for_frame(f, (INPUT_W, INPUT_H))
            img, mask, meta = prepare_lane_input(tf.apply(f))
            reg, _ = infer(img, mask)
            raw, _ = postprocess_onnx_output(reg, conf_threshold=meta["conf"])
            for y, w in pair_widths(raw):
                for b in BANDS:
                    if b[0] <= y < b[1]:
                        per_band[b].append(w)
                        pooled[b].append(w)
                        break

        print(f"\n=== {os.path.basename(video)}  {len(samples)} frames ===")
        print(f"{'depth band':>14} {'n':>7} {'width':>8} {'err':>8} {'scale':>7}")
        print("-" * 48)
        for b in BANDS:
            v = np.asarray(per_band[b], float)
            if len(v) < 20:
                print(f"{b[0]:5.0f}-{b[1]:4.0f} m {len(v):7d} {'--':>8} {'--':>8} {'--':>7}")
                continue
            m = float(v.mean())
            print(f"{b[0]:5.0f}-{b[1]:4.0f} m {len(v):7d} {m:8.2f} "
                  f"{m - TRUE_LANE_M:+8.2f} {TRUE_LANE_M / m:7.3f}")

    print(f"\n=== pooled across clips (truth = {TRUE_LANE_M} m) ===")
    print(f"{'depth band':>14} {'n':>7} {'width':>8} {'err':>8} {'scale':>7} {'std':>7}")
    print("-" * 56)
    scales = []
    for b in BANDS:
        v = np.asarray(pooled[b], float)
        if len(v) < 20:
            continue
        m = float(v.mean())
        scales.append(TRUE_LANE_M / m)
        print(f"{b[0]:5.0f}-{b[1]:4.0f} m {len(v):7d} {m:8.2f} {m - TRUE_LANE_M:+8.2f} "
              f"{TRUE_LANE_M / m:7.3f} {v.std():7.3f}")
    if scales:
        sc = np.asarray(scales)
        print(f"\nscale factor across bands: mean {sc.mean():.3f}  spread {sc.max()-sc.min():.3f}")
        print("small spread -> one constant correction works; large -> depth-dependent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
