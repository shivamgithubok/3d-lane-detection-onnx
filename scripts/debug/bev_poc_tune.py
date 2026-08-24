#!/usr/bin/env python3
"""Tune src.tracking.lane_frame.LaneFrameModel across videos without re-running inference.

Stage 1 (--cache): run the lane backend once per video, pickle raw proposals.
Stage 2 (default):  replay RoadStateEstimator + LaneFrameModel over the cache
                    for many parameter sets and score them.

Scoring balances four things that pull against each other:
  jitter    - frame-to-frame movement of the rendered lane (lower = calmer)
  ego_jit   - shake of the ego car itself
  curv_err  - fidelity to the real road curve, measured against a zero-lag
              (non-causal) reference so that smoothing is not self-penalised.
              Without this term the optimiser simply flattens the road, because
              a dead-straight lane trivially has zero jitter.
  pose_err  - ego still sits where it really is in the lane
"""

import argparse
import itertools
import os
import pickle
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np

from src.inference.lane_preprocess import prepare_lane_input
from src.inference.postprocess import postprocess_onnx_output
from src.tracking.lane_frame import LaneFrameModel
from src.tracking.road_state import RoadStateEstimator
from src.utils.ego_speed import EgoSpeedLog

INPUT_H, INPUT_W = 360, 480
ENGINE_PATH = "models/anchor3dlane_raw.engine"
MODEL_PATH = "models/anchor3dlane_raw.onnx"


def load_backend(force_onnx=False):
    if not force_onnx and os.path.exists(ENGINE_PATH):
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

        print(f"[tune] backend: TensorRT {ENGINE_PATH}")
        return infer

    import onnxruntime as ort

    sess = ort.InferenceSession(MODEL_PATH, providers=ort.get_available_providers())
    print(f"[tune] backend: ONNX {MODEL_PATH}")
    return lambda img, mask: sess.run(None, {"img": img, "mask": mask})

CACHE_DIR = "output/poc_cache"
VIDEOS = [
    "testing_new_videos/GRMN6694_540_nohud.mp4",
    "testing_new_videos/GRMN6695_540_nohud.mp4",
    "testing_new_videos/GRMN6700_540p30_nohud.mp4",
    "data/images/example_1.mp4",
    "data/images/example_3.mp4",
]
# example_traffic.mp4 is excluded: the lane detector returns 0 proposals on every
# frame even at conf 0.05, so it measures detection failure, not render quality.
PROBE_YS = (10.0, 20.0, 40.0)


# --------------------------------------------------------------------------- cache
def cache_video(infer, video, max_frames):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[cache] SKIP (cannot open) {video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    speed_log = EgoSpeedLog.auto_load(video)
    frames, n = [], 0
    while cap.isOpened() and n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        img, mask, meta = prepare_lane_input(cv2.resize(frame, (INPUT_W, INPUT_H)))
        reg, _ = infer(img, mask)
        raw, _ = postprocess_onnx_output(reg, conf_threshold=meta["conf"])
        sp = speed_log.get_mps(n) if speed_log is not None else None
        frames.append(([np.asarray(r, dtype=np.float32) for r in raw], sp))
        n += 1
    cap.release()
    return {"video": video, "fps": float(fps), "frames": frames}


def do_cache(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    infer = load_backend()
    for v in VIDEOS:
        if not os.path.exists(v):
            print(f"[cache] missing {v}")
            continue
        out = os.path.join(CACHE_DIR, os.path.basename(v) + ".pkl")
        if os.path.exists(out) and not args.force:
            print(f"[cache] have {out}")
            continue
        data = cache_video(infer, v, args.max_frames)
        if data is None:
            continue
        with open(out, "wb") as f:
            pickle.dump(data, f)
        print(f"[cache] {v} -> {len(data['frames'])} frames")


# --------------------------------------------------------------------------- replay
def centered_smooth(v, win=31):
    """Non-causal, zero-lag reference: the road shape without noise and without lag.

    Comparing a causal filter against the raw measurement penalises smoothing
    itself. Comparing it against this reference measures real fidelity loss.
    """
    a = np.asarray(v, dtype=float)
    out = np.full(len(a), np.nan)
    half = win // 2
    for i in range(len(a)):
        lo, hi = max(0, i - half), min(len(a), i + half + 1)
        w = a[lo:hi]
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            out[i] = float(np.mean(w))
    return out


def rms_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if int(m.sum()) > 10 else 0.0


def replay(data, params, fallback_speed=24.3):
    dt = 1.0 / max(1e-3, data["fps"])
    est = RoadStateEstimator()
    model = LaneFrameModel()
    for k, v in params.items():
        setattr(model, k, v)

    rendered = {y: [] for y in PROBE_YS}
    pose_s, pose_raw, blank = [], [], 0
    meas40 = []
    for raw, sp in data["frames"]:
        sp = fallback_speed if (sp is None or sp <= 0.5) else sp
        st = est.update(raw, dt=dt, speed_mps=sp)

        # Raw (unsmoothed, unclamped) measurement, for lag + fidelity checks.
        raw_c = None
        if st.left_corridor_3d is not None and st.right_corridor_3d is not None:
            c, _hw = LaneFrameModel._fit_centerline(st.left_corridor_3d, st.right_corridor_3d)
            if c is not None:
                raw_c = np.asarray(c, dtype=float)

        model.update(st.left_corridor_3d, st.right_corridor_3d, sp, dt)
        if not model.valid:
            blank += 1
            for y in PROBE_YS:
                rendered[y].append(np.nan)
            pose_s.append(np.nan)
        else:
            for y in PROBE_YS:
                rendered[y].append(float(model.lane_x(np.array([y]))[0]))
            pose_s.append(model.ego_pose()[0])
        # Pose-removed measured shape at 40 m, where curvature dominates.
        y = 40.0
        meas40.append(np.nan if raw_c is None else raw_c[2] * y ** 2 + raw_c[3] * y ** 3)
        pose_raw.append(np.nan if raw_c is None else -float(raw_c[0]))

    def jit(v):
        a = np.asarray(v, dtype=float)
        d = np.abs(np.diff(a))
        d = d[np.isfinite(d)]
        return float(d.mean()) if len(d) else 0.0

    out = {f"j{int(y)}": jit(rendered[y]) for y in PROBE_YS}
    out["blank"] = blank
    out["ego_jit"] = jit(pose_s)
    out["curv_err"] = rms_diff(rendered[40.0], centered_smooth(meas40))
    out["pose_err"] = rms_diff(pose_s, centered_smooth(pose_raw))
    out["lag"], out["rmse"] = pose_lag(pose_s, pose_raw)
    return out


def pose_lag(smoothed, raw, max_lag=15):
    s = np.asarray(smoothed, float)
    r = np.asarray(raw, float)
    m = np.isfinite(s) & np.isfinite(r)
    if int(m.sum()) < 30:
        return 0.0, 0.0
    s, r = s[m], r[m]
    s = s - s.mean()
    r = r - r.mean()
    if np.std(s) < 1e-6 or np.std(r) < 1e-6:
        return 0.0, 0.0
    best_l, best_c = 0, -2.0
    for l in range(0, max_lag + 1):
        a = s[l:] if l else s
        b = r[: len(r) - l] if l else r
        n = min(len(a), len(b))
        if n < 20:
            break
        c = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if c > best_c:
            best_c, best_l = c, l
    rmse = float(np.sqrt(np.mean((s - r) ** 2)))
    return float(best_l), rmse


# --------------------------------------------------------------------------- sweep
def do_sweep(args):
    caches = []
    for v in VIDEOS:
        p = os.path.join(CACHE_DIR, os.path.basename(v) + ".pkl")
        if os.path.exists(p):
            caches.append(pickle.load(open(p, "rb")))
    if not caches:
        print("no cache; run with --cache first")
        return
    print(f"[sweep] {len(caches)} videos, {sum(len(c['frames']) for c in caches)} frames total\n")

    grid = {
        "pose_alpha": [0.20, 0.35, 0.50],
        "curv_alpha": [0.04, 0.08, 0.15],
        "c2_max": [0.002, 0.004, 0.008],
    }
    keys = list(grid)
    rows = []
    fields = ["j10", "j20", "j40", "blank", "lag", "rmse", "ego_jit", "curv_err", "pose_err"]
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        agg = {k: [] for k in fields}
        for c in caches:
            r = replay(c, params)
            for k in agg:
                agg[k].append(r[k])
        rows.append((params, {k: float(np.mean(v)) for k, v in agg.items()}))

    # Calm road AND steady ego AND still faithful to the real curve.
    # curv_err is what stops the optimiser from just flattening the road.
    def score(m):
        return (m["j40"] * 1.0          # far-field calm
                + m["ego_jit"] * 3.0    # ego car must not shake
                + m["curv_err"] * 1.0   # still follows the real curve
                + m["pose_err"] * 1.0)  # ego still sits where it really is

    rows.sort(key=lambda t: score(t[1]))
    hdr = (f"{'pose_a':>7} {'curv_a':>7} {'c2_max':>8} | {'j10':>7} {'j40':>7} "
           f"{'egojit':>7} {'curverr':>8} {'poserr':>7} {'lag_f':>6} | {'score':>7}")
    print(hdr)
    print("-" * len(hdr))
    for params, m in rows:
        print(f"{params['pose_alpha']:7.2f} {params['curv_alpha']:7.2f} {params['c2_max']:8.3f} | "
              f"{m['j10']:7.4f} {m['j40']:7.4f} {m['ego_jit']:7.4f} {m['curv_err']:8.4f} "
              f"{m['pose_err']:7.4f} {m['lag']:6.2f} | {score(m):7.4f}")

    best = rows[0][0]
    print(f"\nbest: {best}")
    print("\nper-video with best params:")
    print(f"{'video':38s} {'j10':>8} {'j40':>8} {'egojit':>8} {'curverr':>8} {'poserr':>8} {'blank':>6}")
    for c in caches:
        r = replay(c, best)
        print(f"{os.path.basename(c['video']):38s} {r['j10']:8.4f} {r['j40']:8.4f} "
              f"{r['ego_jit']:8.4f} {r['curv_err']:8.4f} {r['pose_err']:8.4f} {r['blank']:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-frames", type=int, default=450)
    args = ap.parse_args()
    if args.cache:
        do_cache(args)
    else:
        do_sweep(args)


if __name__ == "__main__":
    main()
