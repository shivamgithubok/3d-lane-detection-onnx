#!/usr/bin/env python3
"""Per-frame ranging diagnostics: dump CSV, then plot, for one clip.

Runs YOLO + the lane road-state estimator over a video and, for every detection,
records what BOTH ranging paths say:

  * `legacy_*`  -- what `CIPOTracker.project_2d_to_3d_ground` produces today,
                   i.e. the OpenLane P matrix applied to a Garmin frame.
  * `geom_*`    -- gated ground-plane back-projection with the measured
                   calibration from scripts/calibrate_from_video.py.
  * `kf_*`      -- the BEV Kalman output, so filter lag/overshoot is visible
                   separately from measurement noise.

The point is to make the three separable. A symptom that shows up in `geom` and
`kf` but not in the measurement is filter tuning; one in `legacy` but not `geom`
is calibration; one in all three is the detector.

Usage
-----
    python scripts/debug/range_diagnostics.py testing_new_videos/GRMN6694_540_nohud.mp4 \
        --frames 400 --out output/diag
    python scripts/debug/range_diagnostics.py CLIP.mp4 --plot-track 7

Interpretation (as specified in the audit brief)
------------------------------------------------
  slowly wandering baseline        -> scale drift          (cause B)
  high-freq spikes, stable mean    -> bbox jitter          (cause H / J)
  constant lateral bias, straight  -> extrinsics / origin  (cause D / E)
  lateral magnitude too small      -> calibration          (cause A / C)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.object_detector import OfflineYOLOVehicleDetector      # noqa: E402
from src.tracking.bev_kalman import BevTracker                            # noqa: E402
from src.tracking.lane_assignment import LaneModel, assign_lane, place_in_lane, lane_index_raw  # noqa: E402
from src.tracking.lane_frame import LaneFrameModel                        # noqa: E402
from src.tracking.road_state import RoadStateEstimator                    # noqa: E402
from src.utils.calibration import P_final                                 # noqa: E402
from src.utils.camera_transform import CameraTransform                    # noqa: E402
from src.utils.ego_speed import EgoSpeedLog                               # noqa: E402
from src.utils.ground_calib import GroundCalibration, bbox_ground_point   # noqa: E402
from src.utils.ground_gate import measure_ground, GateResult              # noqa: E402

FIELDS = [
    "frame_id", "t_s", "track_id", "class", "conf",
    "u1", "v1", "u2", "v2", "bc_u", "bc_v",
    "dv_below_horizon",
    "legacy_Z_m", "legacy_X_m",
    "geom_Z_m", "geom_X_m", "geom_sigma_Z_m", "geom_sigma_X_m", "gate",
    "kf_x_m", "kf_y_m", "kf_vx", "kf_vy", "kf_confirmed",
    "lane_index", "lane_index_raw", "lane_offset_m", "lane_center_x_at_y",
    "bev_render_x", "bev_render_z",
    "ego_speed_mps", "road_status",
]


def legacy_ground(P, u_model, v_model):
    """Verbatim copy of CIPOTracker.project_2d_to_3d_ground, clamp included.

    Copied rather than imported so the harness keeps reporting the original
    behaviour after that method is fixed, which is what makes before/after
    plots meaningful.
    """
    denom_y = (P[2, 1] * v_model - P[1, 1])
    Y = 50.0 if abs(denom_y) < 1e-4 else float(P[1, 3] / denom_y)
    Y = max(1.0, min(100.0, Y))          # <-- the clamp that reports 1.0 m
    X = float((Y * (P[2, 1] * u_model - P[0, 1])) / P[0, 0])
    return X, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--out", default="output/diag")
    ap.add_argument("--plot-track", type=int, default=None,
                    help="track_id to plot; default = longest-lived track")
    ap.add_argument("--no-lanes", action="store_true",
                    help="skip the lane engine (much faster; no lane columns)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    csv_path = os.path.join(args.out, f"{stem}_diag.csv")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    calib = GroundCalibration.for_video(args.video, (h, w))
    print(f"clip   {stem}  {w}x{h} @ {fps:.1f}")
    print(f"calib  {calib.source}: f={calib.f_px:.0f} v_vp={calib.v_vp:.1f} "
          f"u_vp={calib.u_vp:.1f} h={calib.cam_height_m:.3f} h*f={calib.hf:.0f}")
    if not calib.source.startswith("measured"):
        print("       WARNING: fallback calibration. Run calibrate_from_video.py first.")

    P = np.asarray(P_final, dtype=np.float64)
    detector = OfflineYOLOVehicleDetector(model_path="models/yolov8n.pt", conf_thresh=0.22)
    road = None if args.no_lanes else RoadStateEstimator()
    lane_frame = LaneFrameModel()
    tracker = BevTracker()
    speed_log = EgoSpeedLog.auto_load(args.video)

    rows = []
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        fid = args.start + i
        t_s = fid / fps
        speed = speed_log.get_mps(fid) if speed_log is not None else None
        ft = CameraTransform.for_frame(frame, (480, 360))

        # Lane geometry. The lane engine needs the TensorRT engine; when it is
        # unavailable we still record everything else, with empty lane columns.
        status, lane_model = "SKIPPED", None
        if road is not None:
            try:
                rs = road.update([], dt=1.0 / fps, speed_mps=speed)
                status = rs.status
                lane_frame.update(rs.left_corridor_3d, rs.right_corridor_3d,
                                  speed_mps=speed, dt=1.0 / fps)
                lane_model = LaneModel.from_lane_frame(lane_frame)
            except Exception as exc:                       # noqa: BLE001
                status = f"ERR:{type(exc).__name__}"

        dets = detector.detect(frame) or []

        # Build gated geometric measurements for the BEV filter.
        meas, per_det = [], []
        for d in dets:
            bb = d["bbox"]
            bc_u, bc_v = bbox_ground_point(bb)
            gm = measure_ground(bb, calib, d.get("class", "car"),
                               frame_shape=(h, w))
            uv = ft.source_to_model(np.array([[bc_u, bc_v]], dtype=np.float64))[0]
            lx, lz = legacy_ground(P, uv[0], uv[1])
            per_det.append((d, bc_u, bc_v, gm, lx, lz))
            if gm.valid:
                meas.append({"x": gm.x_m, "y": gm.y_m, "R": gm.R,
                             "label": d.get("class", "car"),
                             "conf": d.get("conf", 1.0), "bbox": list(bb),
                             "det_id": int(d.get("track_id", -1))})

        tracker.update(meas, 1.0 / fps, ego_speed_mps=speed)
        by_id = {t.track_id: t for t in tracker.tracks.values()}

        for d, bc_u, bc_v, gm, lx, lz in per_det:
            tid = int(d.get("track_id", -1))
            tr = by_id.get(tid)
            kx = ky = kvx = kvy = ""
            if tr is not None:
                kx, ky, kvx, kvy = tr.x[0], tr.x[1], tr.x[2], tr.x[3]
            li = lir = loff = lcx = rx = rz = ""
            src_x = tr.x[0] if tr is not None else (gm.x_m if gm.valid else None)
            src_y = tr.x[1] if tr is not None else (gm.y_m if gm.valid else None)
            if lane_model is not None and src_x is not None:
                li, loff = assign_lane(src_x, src_y, lane_model)
                lir = lane_index_raw(src_x, src_y, lane_model)
                lcx = float(lane_model.center_x(src_y))
                rx, rz = place_in_lane(src_x, src_y, lane_model, li)
            elif src_x is not None:
                rx, rz = src_x, src_y
            rows.append({
                "frame_id": fid, "t_s": round(t_s, 4), "track_id": tid,
                "class": d.get("class", "car"), "conf": round(float(d.get("conf", 0)), 4),
                "u1": bb[0], "v1": bb[1], "u2": bb[2], "v2": bb[3],
                "bc_u": round(bc_u, 2), "bc_v": round(bc_v, 2),
                "dv_below_horizon": round(bc_v - calib.v_vp, 2),
                "legacy_Z_m": round(lz, 3), "legacy_X_m": round(lx, 3),
                "geom_Z_m": round(gm.y_m, 3) if gm.valid else "",
                "geom_X_m": round(gm.x_m, 3) if gm.valid else "",
                "geom_sigma_Z_m": round(gm.sigma_y_m, 3) if gm.valid else "",
                "geom_sigma_X_m": round(gm.sigma_x_m, 3) if gm.valid else "",
                "gate": gm.result.value,
                "kf_x_m": round(kx, 3) if kx != "" else "",
                "kf_y_m": round(ky, 3) if ky != "" else "",
                "kf_vx": round(kvx, 3) if kvx != "" else "",
                "kf_vy": round(kvy, 3) if kvy != "" else "",
                "kf_confirmed": int(tr.confirmed) if tr is not None else "",
                "lane_index": li, "lane_index_raw": lir,
                "lane_offset_m": round(loff, 3) if loff != "" else "",
                "lane_center_x_at_y": round(lcx, 3) if lcx != "" else "",
                "bev_render_x": round(rx, 3) if rx != "" else "",
                "bev_render_z": round(rz, 3) if rz != "" else "",
                "ego_speed_mps": round(speed, 3) if speed else "",
                "road_status": status,
            })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.frames} frames, {len(rows)} detections")

    cap.release()
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")

    _summarise(rows, calib)
    _plot(rows, args.plot_track, args.out, stem, calib)


def _summarise(rows, calib):
    if not rows:
        return
    gates = {}
    for r in rows:
        gates[r["gate"]] = gates.get(r["gate"], 0) + 1
    print("\ngate outcomes:")
    for k, v in sorted(gates.items(), key=lambda t: -t[1]):
        print(f"  {k:22s} {v:6d}  ({100.0*v/len(rows):5.1f}%)")

    both = [r for r in rows if r["geom_Z_m"] != ""]
    if both:
        lz = np.array([float(r["legacy_Z_m"]) for r in both])
        gz = np.array([float(r["geom_Z_m"]) for r in both])
        lx = np.array([float(r["legacy_X_m"]) for r in both])
        gx = np.array([float(r["geom_X_m"]) for r in both])
        print(f"\nlegacy vs geometric, on the {len(both)} detections both accept:")
        print(f"  range  legacy/geom ratio  median {np.median(lz/gz):.3f}"
              f"  (p10 {np.percentile(lz/gz,10):.3f}, p90 {np.percentile(lz/gz,90):.3f})")
        nz = np.abs(gx) > 0.5
        if nz.any():
            print(f"  lateral legacy/geom ratio median {np.median(lx[nz]/gx[nz]):.3f}"
                  f"  <-- <1 means the legacy path pulls objects toward the centreline")
        clamped = int(np.sum(np.array([float(r['legacy_Z_m']) for r in rows]) <= 1.001))
        print(f"  legacy range pinned at the 1.0 m clamp floor: {clamped} rows "
              f"({100.0*clamped/len(rows):.1f}%)")
        print(f"  legacy range saturation ceiling for this camera: "
              f"{calib.hf/max(1e-6, calib.v_vp-302.8):.0f} m")


def _plot(rows, want_tid, out_dir, stem, calib):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots")
        return
    counts = {}
    for r in rows:
        t = r["track_id"]
        if t and int(t) > 0:
            counts[int(t)] = counts.get(int(t), 0) + 1
    if not counts:
        print("\nno tracked objects to plot")
        return
    tid = want_tid if want_tid is not None else max(counts, key=counts.get)
    sub = [r for r in rows if r["track_id"] and int(r["track_id"]) == tid]
    if len(sub) < 10:
        print(f"\ntrack {tid} has only {len(sub)} frames; skipping plots")
        return
    t = np.array([float(r["t_s"]) for r in sub])

    def col(name):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in sub])

    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    gz, lz, kz = col("geom_Z_m"), col("legacy_Z_m"), col("kf_y_m")
    sz = col("geom_sigma_Z_m")
    ax[0].plot(t, lz, ".-", lw=1, ms=3, label="legacy (OpenLane P)", color="#d62728")
    ax[0].plot(t, gz, ".-", lw=1, ms=3, label="ground-plane (measured calib)", color="#1f77b4")
    ax[0].plot(t, kz, "-", lw=2, label="BEV Kalman", color="#2ca02c")
    ok = ~np.isnan(gz) & ~np.isnan(sz)
    ax[0].fill_between(t[ok], (gz - 2 * sz)[ok], (gz + 2 * sz)[ok],
                       alpha=0.18, color="#1f77b4", label=r"$\pm2\sigma$ from $d^2/hf$")
    ax[0].axhline(calib.hf / max(1e-6, calib.v_vp - 302.8), ls=":", color="#d62728",
                  label="legacy saturation ceiling")
    ax[0].set_ylabel("forward range (m)")
    ax[0].set_title(f"{stem} — track {tid}: range")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    gx, lx, kx = col("geom_X_m"), col("legacy_X_m"), col("kf_x_m")
    ax[1].plot(t, lx, ".-", lw=1, ms=3, label="legacy", color="#d62728")
    ax[1].plot(t, gx, ".-", lw=1, ms=3, label="ground-plane", color="#1f77b4")
    ax[1].plot(t, kx, "-", lw=2, label="BEV Kalman", color="#2ca02c")
    rxc = col("bev_render_x")
    if not np.all(np.isnan(rxc)):
        ax[1].plot(t, rxc, "--", lw=1.2, label="lane-relative render x", color="#9467bd")
    ax[1].axhline(0, color="k", lw=0.8)
    for k in (-1, 1):
        ax[1].axhline(k * 1.85, ls=":", color="gray", lw=0.8)
    ax[1].set_ylabel("lateral offset (m)")
    ax[1].set_title("lateral — dotted lines are the ego-lane edges")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    dv = col("dv_below_horizon")
    ax[2].plot(t, dv, ".-", lw=1, ms=3, color="#ff7f0e", label="rows below true horizon")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].axhline(calib.v_vp - 302.8, ls=":", color="#d62728",
                  label="legacy horizon error (13.2 px)")
    ax[2].set_ylabel("v - v_vp (px)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_title("bbox bottom edge vs horizon — the quantity range is 1/x in")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"{stem}_track{tid}_range.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # Range error growth: what a pixel is worth, measured vs legacy.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    zz = np.linspace(5, 100, 200)
    ax[0].plot(zz, zz ** 2 / calib.hf, label=f"measured  h*f={calib.hf:.0f}")
    ax[0].plot(zz, zz ** 2 / 950.5, ls="--", label="legacy h*f=950.5")
    ax[0].set_xlabel("range (m)")
    ax[0].set_ylabel("metres per pixel of bottom-edge jitter")
    ax[0].set_title(r"$d^2/hf$ range sensitivity")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    true_z = np.linspace(8, 100, 200)
    v = calib.v_vp + calib.hf / true_z
    rep = 950.5 / np.maximum(1e-6, v - 302.8)
    ax[1].plot(true_z, true_z, "k:", label="ideal")
    ax[1].plot(true_z, np.clip(rep, 1.0, 100.0), color="#d62728",
               label="what legacy reports")
    ax[1].set_xlabel("true range (m)")
    ax[1].set_ylabel("reported range (m)")
    ax[1].set_title("legacy range compression")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(out_dir, f"{stem}_sensitivity.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
