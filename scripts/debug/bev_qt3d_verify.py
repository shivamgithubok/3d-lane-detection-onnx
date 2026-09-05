#!/usr/bin/env python3
"""Exercise the real Qt Quick 3D BEV from cached detections.

Replays scripts/debug/bev_poc_tune.py's detection cache through the production
RoadStateEstimator and BevQuick3DWidget, so the actual QML scene and the real
payload builders are driven without needing CUDA in the Qt process.

  --shots    save PNG grabs at chosen frames (visual check)
  --metrics  run the whole clip and print render-stability numbers (regression)

Usage:
  python scripts/debug/bev_poc_tune.py --cache          # once, to build the cache
  python scripts/debug/bev_qt3d_verify.py --metrics
  python scripts/debug/bev_qt3d_verify.py --shots 60 150 240 380
"""

import argparse
import json
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("QSG_RHI_BACKEND", "opengl")

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtWidgets import QApplication

from src.tracking.road_state import RoadStateEstimator
from src.ui.bev_quick3d import BevQuick3DWidget

CACHE_DIR = "output/poc_cache"
OUT_DIR = "output/qt3d_frames"
PROBE_YS = np.array([10.0, 20.0, 40.0])


def load_cache(name):
    path = os.path.join(CACHE_DIR, name + ".pkl")
    if not os.path.exists(path):
        print(f"missing cache {path}; run: python scripts/debug/bev_poc_tune.py --cache")
        return None
    return pickle.load(open(path, "rb"))


def jitter(vals):
    a = np.asarray(vals, dtype=float)
    d = np.abs(np.diff(a))
    d = d[np.isfinite(d)]
    return (float(d.mean()), float(d.max())) if len(d) else (0.0, 0.0)


def run_metrics(widget, data, fallback_speed):
    dt = 1.0 / max(1e-3, data["fps"])
    est = RoadStateEstimator()
    lf = widget.lane_frame
    root = widget.rootObject()
    lane_x = {y: [] for y in PROBE_YS}
    ego_x, dist, blank, held = [], [], 0, 0
    seg_counts = {"dash": [], "edge": [], "lane": [], "corr": []}

    for raw, sp in data["frames"]:
        sp = fallback_speed if (sp is None or sp <= 0.5) else sp
        st = est.update(raw, dt=dt, speed_mps=sp)
        widget.update_bev_data(st.visual_lanes, [], st.status,
                               st.left_corridor_3d, st.right_corridor_3d,
                               speed_mps=sp, dt=dt)
        if not lf.valid:
            blank += 1
            for y in PROBE_YS:
                lane_x[y].append(np.nan)
            ego_x.append(np.nan)
        else:
            held += 1 if lf.held else 0
            for y in PROBE_YS:
                lane_x[y].append(float(lf.lane_x(np.array([y]))[0]))
            ego_x.append(lf.ego_pose()[0])
        dist.append(lf.distance)
        for key, prop in (("dash", "dashJson"), ("edge", "edgeJson"),
                          ("lane", "laneJson"), ("corr", "corridorJson")):
            try:
                seg_counts[key].append(len(json.loads(root.property(prop) or "[]")))
            except Exception:
                seg_counts[key].append(0)

    n = len(data["frames"])
    print(f"\n=== {data['video']} | {n} frames | production BevQuick3DWidget ===\n")
    for y in PROBE_YS:
        m, mx = jitter(lane_x[y])
        print(f"  lane jitter @{int(y):2d}m   mean {m:.4f} m   max {mx:.4f} m")
    m, mx = jitter(ego_x)
    print(f"  ego lateral jitter    mean {m:.4f} m   max {mx:.4f} m")
    ex = [v for v in ego_x if np.isfinite(v)]
    if ex:
        print(f"  ego offset range      [{min(ex):+.2f}, {max(ex):+.2f}] m")
    ds = np.diff(dist)
    print(f"  road scroll / frame   mean {ds.mean():.4f} m   (must track v*dt)")
    print(f"  road blank frames     {blank} / {n}      held frames {held}")
    for k, v in seg_counts.items():
        v = np.asarray(v, float)
        ch = np.abs(np.diff(v))
        print(f"  {k:5s} segs mean {v.mean():5.2f}  count changed on {int((ch>0).sum()):3d}/{n-1} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="GRMN6695_540_nohud.mp4")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--shots", type=int, nargs="*", default=None)
    ap.add_argument("--fallback-speed", type=float, default=24.3)
    args = ap.parse_args()
    if not args.metrics and args.shots is None:
        args.shots = [60, 150, 240, 300, 380]

    data = load_cache(args.video)
    if data is None:
        return 1
    dt = 1.0 / max(1e-3, data["fps"])
    os.makedirs(OUT_DIR, exist_ok=True)

    QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    app = QApplication(sys.argv)
    w = BevQuick3DWidget()
    w.resize(640, 820)
    w.show()

    state = {"i": 0, "warm": 25, "saved": 0, "done": False}
    est = RoadStateEstimator()
    targets = sorted(set(args.shots or []))

    def tick():
        # Let the GLB load and the scene graph settle before stepping.
        if state["warm"] > 0:
            state["warm"] -= 1
            return
        if state["done"]:
            return
        if args.metrics:
            state["done"] = True
            run_metrics(w, data, args.fallback_speed)
            app.quit()
            return

        i = state["i"]
        if i >= len(data["frames"]):
            print(f"done, saved {state['saved']} frames -> {OUT_DIR}")
            app.quit()
            return
        raw, sp = data["frames"][i]
        sp = args.fallback_speed if (sp is None or sp <= 0.5) else sp
        st = est.update(raw, dt=dt, speed_mps=sp)
        w.update_bev_data(st.visual_lanes, [], st.status, st.left_corridor_3d,
                          st.right_corridor_3d, speed_mps=sp, dt=dt)
        state["i"] += 1

        if i in targets:
            root = w.rootObject()
            img = w.grabFramebuffer()
            if img.isNull():
                img = w.grab().toImage()
            out = os.path.join(OUT_DIR, f"{args.video}_{i:04d}.png")
            img.save(out)
            state["saved"] += 1
            print(f"frame {i:4d} {st.status:9s} egoX={root.property('egoX'):+.2f} "
                  f"yaw={root.property('egoYawDeg'):+.1f} held={root.property('laneHeld')} "
                  f"-> {out}")

    t = QTimer()
    t.setInterval(16)
    t.timeout.connect(tick)
    t.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
