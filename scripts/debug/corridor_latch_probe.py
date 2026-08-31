#!/usr/bin/env python3
"""Diagnose ego-corridor latching onto an adjacent lane, and missing-lane frames.

Runs the production preprocess + TensorRT + RoadStateEstimator path. For every
sampled frame it records:

  raw_n        proposals after NMS
  pair source  width_pair / hold_* / onesided_* / none
  center_m     0.5*(left_x + right_x) of the *emitted* corridor
  contains0    whether left_x < 0 < right_x (true ego occupancy in camera frame)
  better_pair  a width-valid pair whose |center| is smaller than the chosen one

A corridor that does not contain X=0 is sitting in an adjacent lane (or the
car is mid-straddle). A chosen pair that loses to a better-centered candidate
is a scoring bug in find_ego_lanes.

Usage:
  python scripts/debug/corridor_latch_probe.py --video testing_new_videos/ADAS3_540.mp4
"""

from __future__ import annotations

import argparse
import collections
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
from src.utils.drivable_area import (
    find_ego_lanes,
    lane_mean_x,
    pair_gap_m,
    pair_occupancy_tier,
    ego_pair_score,
)

INPUT_H, INPUT_W = 360, 480
ENGINE_PATH = "models/anchor3dlane_raw.engine"
LATCH_CENTER_M = 1.6  # |center| above this => corridor is not the ego lane


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


def all_width_pairs(proposals):
    lanes = []
    for lane in proposals:
        mx = lane_mean_x(lane)
        if mx is not None:
            lanes.append((mx, lane))
    lanes.sort(key=lambda t: t[0])
    out = []
    for i in range(len(lanes)):
        for j in range(i + 1, len(lanes)):
            ml, left = lanes[i]
            mr, right = lanes[j]
            gap = pair_gap_m(left, right)
            if gap is None:
                continue
            if not (cfg.EGO_LANE_WIDTH_MIN_M <= gap <= cfg.EGO_LANE_WIDTH_MAX_M):
                continue
            center = 0.5 * (ml + mr)
            contains0 = ml < 0.0 < mr
            tier = pair_occupancy_tier(ml, mr)
            score = ego_pair_score(gap, center, cfg.EGO_LANE_WIDTH_TARGET_M)
            out.append({"ml": ml, "mr": mr, "gap": gap, "center": center,
                        "contains0": contains0, "tier": tier, "score": score})
    return out


def dump_frame(path, frame, st, pairs, note):
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.putText(vis, note, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        vis,
        f"status={st.status} src={st.source} nvis={len(st.visual_lanes)}",
        (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    y = 108
    for p in pairs[:6]:
        mark = "Oc" if p.get("tier") == 0 else ("C0" if p["contains0"] else "  ")
        line = (f"{mark} L={p['ml']:+.2f} R={p['mr']:+.2f} "
                f"c={p['center']:+.2f} w={p['gap']:.2f} sc={p['score']:.2f}")
        cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        y += 22
    cv2.imwrite(path, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="testing_new_videos/ADAS3_540.mp4")
    ap.add_argument("--max-frames", type=int, default=1800)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--dump-dir", default="output/diag/adas3_corridor")
    args = ap.parse_args()

    infer = load_trt()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / max(1e-3, fps)
    os.makedirs(args.dump_dir, exist_ok=True)

    est = RoadStateEstimator()
    lf = LaneFrameModel()
    counts = collections.Counter()
    miss_raw = 0
    miss_rs = 0
    latch = 0
    scoring_wrong = 0
    dumped = 0
    n = 0
    luma = []
    status_hist = collections.Counter()
    src_hist = collections.Counter()
    centers = []
    streaks_miss = []
    cur_miss = 0

    idx = -1
    while n < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.stride != 0:
            continue
        n += 1

        tf = CameraTransform.for_frame(frame, (INPUT_W, INPUT_H))
        resized = tf.apply(frame)
        img, mask, meta = prepare_lane_input(resized)
        luma.append(float(meta.get("luma", np.nan)) if isinstance(meta, dict) else np.nan)
        reg, _ = infer(img, mask)
        raw, scores = postprocess_onnx_output(reg, conf_threshold=meta["conf"])
        st = est.update(raw, dt=dt, speed_mps=None)
        lf.update(st.left_corridor_3d, st.right_corridor_3d, speed_mps=None, dt=dt)

        counts["frames"] += 1
        counts["raw_lanes"] += len(raw)
        if len(raw) == 0:
            miss_raw += 1
            cur_miss += 1
        else:
            if cur_miss:
                streaks_miss.append(cur_miss)
            cur_miss = 0
        status_hist[st.status] += 1
        src_hist[st.source] += 1

        cand_l, cand_r = find_ego_lanes(raw)
        pairs = all_width_pairs(raw)
        containing = [p for p in pairs if p["contains0"]]
        best_c0 = min(containing, key=lambda p: abs(p["center"])) if containing else None

        chosen_c = None
        contains0 = False
        if st.ego_left is not None and st.ego_right is not None:
            ml = lane_mean_x(st.ego_left)
            mr = lane_mean_x(st.ego_right)
            if ml is not None and mr is not None:
                chosen_c = 0.5 * (ml + mr)
                contains0 = ml < 0.0 < mr
                centers.append(chosen_c)
                if abs(chosen_c) >= LATCH_CENTER_M or not contains0:
                    latch += 1
                    if dumped < 12:
                        dump_frame(
                            os.path.join(args.dump_dir, f"latch_{idx:05d}.png"),
                            frame, st, pairs,
                            f"LATCH f={idx} c={chosen_c:+.2f} contains0={contains0}",
                        )
                        dumped += 1
                if best_c0 is not None and abs(best_c0["center"]) + 0.4 < abs(chosen_c):
                    scoring_wrong += 1

        if st.left_corridor_3d is None:
            miss_rs += 1
            if dumped < 24 and len(raw) == 0:
                dump_frame(
                    os.path.join(args.dump_dir, f"miss_{idx:05d}.png"),
                    frame, st, pairs,
                    f"MISS f={idx} conf={meta.get('conf')} luma={meta.get('luma')}",
                )
                dumped += 1

        if n <= 5 or n % 200 == 0:
            print(f"  f={idx:5d} raw={len(raw):2d} {st.status:9s} {st.source:16s} "
                  f"c={chosen_c if chosen_c is not None else float('nan'):+.2f} "
                  f"c0={contains0}")

    if cur_miss:
        streaks_miss.append(cur_miss)
    cap.release()

    print(f"\n=== {os.path.basename(args.video)}  {n} sampled frames "
          f"(stride={args.stride}, ~{n * args.stride / fps:.1f}s) ===\n")
    print(f"  mean raw lanes / frame     {counts['raw_lanes'] / max(1, n):.2f}")
    print(f"  frames with 0 raw lanes    {miss_raw} / {n}  ({100 * miss_raw / max(1, n):.1f}%)")
    print(f"  frames with no corridor    {miss_rs} / {n}  ({100 * miss_rs / max(1, n):.1f}%)")
    print(f"  adjacent-latch frames      {latch} / {n}  "
          f"(|center|>= {LATCH_CENTER_M} m or pair does not contain X=0)")
    print(f"  scoring picked worse pair  {scoring_wrong} / {n}")
    print(f"  status                     {dict(status_hist)}")
    print(f"  pair source                {dict(src_hist)}")
    if centers:
        a = np.asarray(centers)
        print(f"  corridor center            mean {a.mean():+.2f}  "
              f"p50 {np.median(a):+.2f}  p95 {np.percentile(np.abs(a), 95):.2f}  "
              f"max |c| {np.abs(a).max():.2f}")
    if luma:
        lv = np.asarray([x for x in luma if np.isfinite(x)])
        if len(lv):
            print(f"  model luma (HSV-V)         mean {lv.mean():.1f}  "
                  f"dark(<{cfg.DARK_LUMA_MAX}) {(lv < cfg.DARK_LUMA_MAX).mean() * 100:.1f}%")
    if streaks_miss:
        print(f"  zero-lane streaks          n={len(streaks_miss)}  "
              f"max={max(streaks_miss)} samples  "
              f"mean={np.mean(streaks_miss):.1f}")
    print(f"\n  dumps -> {args.dump_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
