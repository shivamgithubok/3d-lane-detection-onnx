#!/usr/bin/env python3
"""Self-calibrate a dashcam from its own footage. No calibration target needed.

Recovers everything ground-plane ranging needs, in the order that each quantity
becomes observable:

  1. v_vp, u_vp  -- vanishing point, from the intersection of lane markings.
     Absorbs pitch and yaw exactly, so neither has to be measured separately.

  2. h  -- camera height, from lane-marking separation versus image row, given a
     known lane width (3.7 m US / 3.5 m EU).  Because
         W = h * (u_right - u_left) / (v - v_vp)
     this needs no focal length at all.  Fitting separation-vs-row as a straight
     line also re-derives v_vp as its zero crossing, giving an independent check
     on step 1 that shares none of its assumptions.

  3. f  -- focal length, from the apparent motion of static road texture against
     a known ego speed.  For a static ground point,
         1/(v_t - v_vp) = (y_0 - s_t) / (h*f)
     is linear in distance travelled s_t with slope -1/(h*f).  Fitting all
     tracked points jointly (shared slope, per-track intercept) recovers h*f,
     and f = h*f / h.

Step 3 is the only one that needs the speed log, and it is the only one that
affects forward range.  Steps 1 and 2 fully determine *lateral* offset, which is
why lateral placement can be made correct even on a clip with no speed data.

Usage
-----
    python scripts/calibrate_from_video.py testing_new_videos/GRMN6694_540_nohud.mp4
    python scripts/calibrate_from_video.py CLIP.mp4 --lane-width 3.5 --write

Writes models/calib/<stem>.json, which GroundCalibration.for_video picks up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.ground_calib import GroundCalibration  # noqa: E402
from src.utils.self_calib import generic_calib  # noqa: E402

MPH_TO_MPS = 0.44704


def _hough_xyxy(line):
    """OpenCV HoughLinesP shape varies (N,1,4) vs (N,4) vs ragged."""
    return [int(v) for v in np.asarray(line).reshape(-1)[:4]]


# --------------------------------------------------------------- step 1: VP
def estimate_vanishing_point(cap, roi_top_frac=0.45, roi_bot_frac=0.92,
                             n_frames=50, stride=10, start_frame=30):
    """Median intersection of near-field lane-marking lines."""
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    r0, r1 = int(roi_top_frac * h_img), int(roi_bot_frac * h_img)
    min_len = max(20, int(0.08 * w_img))
    pts = []
    for i in range(n_frames):
        fi = start_frame + i * stride
        if n_total > 0 and fi >= n_total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(g)
        mask[r0:r1, :] = 255
        edges = cv2.bitwise_and(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150), mask)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                                minLineLength=min_len, maxLineGap=12)
        if lines is None:
            continue
        segs = []
        for line in lines:
            x1, y1, x2, y2 = _hough_xyxy(line)
            if abs(y2 - y1) < 8:
                continue
            s = (x2 - x1) / float(y2 - y1)     # u = s*v + b
            if abs(s) > 3.5:
                continue
            segs.append((s, x1 - s * y1))
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                s1, b1 = segs[a]
                s2, b2 = segs[b]
                if abs(s1 - s2) < 0.25:
                    continue
                v = (b2 - b1) / (s1 - s2)
                u = s1 * v + b1
                if 0.25 * h_img < v < 0.88 * h_img and 0.15 * w_img < u < 0.85 * w_img:
                    pts.append((u, v))
    if len(pts) < 8:
        return None
    arr = np.array(pts)
    u_vp, v_vp = np.median(arr, axis=0)
    return {
        "u_vp": float(u_vp),
        "v_vp": float(v_vp),
        "n": len(arr),
        "v_iqr": [float(np.percentile(arr[:, 1], 25)),
                  float(np.percentile(arr[:, 1], 75))],
        "u_iqr": [float(np.percentile(arr[:, 0], 25)),
                  float(np.percentile(arr[:, 0], 75))],
    }


# ----------------------------------------------- step 2: height + VP re-check
def estimate_height(cap, v_vp_hint, lane_width_m=3.7, n_frames=40, stride=15):
    """Fit lane-marking separation vs row: slope gives h, zero gives v_vp."""
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = [int(f * h_img) for f in (0.755, 0.795, 0.835, 0.875)]
    samples = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30 + i * stride)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for row in rows:
            if row < 2 or row >= h_img - 3:
                continue
            strip = g[row - 2:row + 3, :].mean(axis=0)
            thr = strip.mean() + 1.4 * strip.std()
            peaks = [u for u in range(int(0.15 * len(strip)), int(0.92 * len(strip)))
                     if strip[u] > thr and strip[u] >= strip[max(0, u - 3):u + 4].max()]
            groups = []
            for u in peaks:
                if groups and u - groups[-1][-1] <= 6:
                    groups[-1].append(u)
                else:
                    groups.append([u])
            cents = [float(np.mean(gg)) for gg in groups]
            # The ego lane pair straddles the vanishing-point column.
            left = [c for c in cents if c < v_vp_hint[0]]
            right = [c for c in cents if c > v_vp_hint[0]]
            if not left or not right:
                continue
            samples.append((row, max(left), min(right)))
    if len(samples) < 8:
        return None
    arr = np.array(samples, dtype=np.float64)
    # Robust line fit of separation vs row.
    v, sep = arr[:, 0], arr[:, 2] - arr[:, 1]
    keep = np.ones(len(v), bool)
    for _ in range(3):
        k, c = np.polyfit(v[keep], sep[keep], 1)
        resid = np.abs(sep - (k * v + c))
        keep = resid < max(3.0, 2.5 * np.median(resid))
        if keep.sum() < 6:
            break
    k, c = np.polyfit(v[keep], sep[keep], 1)
    if k <= 0:
        return None
    return {
        "cam_height_m": float(lane_width_m / k),
        "v_vp": float(-c / k),
        "slope_px_per_row": float(k),
        "n": int(keep.sum()),
        "rms_px": float(np.sqrt(np.mean((sep[keep] - (k * v[keep] + c)) ** 2))),
    }


# ------------------------------------------------------------- step 3: h*f
def estimate_hf(cap, v_vp, speed_mps, baseline=20, stride=20, fps=30.0):
    """Global fit of 1/(v-v_vp) vs distance travelled over many static tracks."""
    dt = 1.0 / float(fps)
    lk = dict(winSize=(25, 25), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005))
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    X, Y = [], []
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f0 in range(250, min(n_total - baseline - 1, 6000), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames, spd = [], []
        for k in range(baseline + 1):
            ok, im = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            idx = min(f0 + k, len(speed_mps) - 1)
            spd.append(float(speed_mps[idx]))
        if len(frames) < baseline + 1 or min(spd) < 8.0:
            continue
        s = np.concatenate([[0.0], np.cumsum(np.array(spd[:-1]) * dt)])
        mask = np.zeros_like(frames[0])
        mask[int(0.655 * h_img):int(0.845 * h_img),
             int(0.12 * w_img):int(0.90 * w_img)] = 255
        p = cv2.goodFeaturesToTrack(frames[0], mask=mask, maxCorners=400,
                                    qualityLevel=0.02, minDistance=8)
        if p is None:
            continue
        tr, alive = [p], np.ones(len(p), bool)
        for k in range(1, len(frames)):
            pn, st, _ = cv2.calcOpticalFlowPyrLK(frames[k - 1], frames[k], tr[-1], None, **lk)
            alive &= (st[:, 0] == 1)
            tr.append(pn)
        T = np.stack([t[:, 0, :] for t in tr])
        for i in range(T.shape[1]):
            if not alive[i]:
                continue
            a = T[:, i, 1] - v_vp
            if a.min() < 8 or np.any(np.diff(a) <= 0.05):
                continue                       # must be static ground approaching
            if a.max() / a.min() < 1.5:
                continue                       # needs depth span to condition the fit
            inv = 1.0 / a
            A = np.polyfit(s, inv, 1)
            if A[0] >= 0:
                continue
            if np.max(np.abs(inv - np.polyval(A, s))) / (inv.max() - inv.min()) > 0.03:
                continue                       # reject anything the model misfits
            X.append(s - s.mean())
            Y.append(inv - inv.mean())
    if not X:
        return None
    X, Y = np.concatenate(X), np.concatenate(Y)
    slope = float(np.sum(X * Y) / np.sum(X * X))
    if slope >= 0:
        return None
    hf = -1.0 / slope
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(400):
        idx = rng.integers(0, len(X), len(X))
        sl = np.sum(X[idx] * Y[idx]) / np.sum(X[idx] * X[idx])
        if sl < 0:
            bs.append(-1.0 / sl)
    return {
        "hf": float(hf),
        "n_obs": int(len(X)),
        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    }


def load_speed(video_path, fps):
    """Ego speed in m/s per frame from the sidecar HUD-OCR JSON, if present."""
    for suffix in ("_speed.json", ".speed.json"):
        base = os.path.splitext(video_path)[0]
        for cand in (base + suffix, base.replace("_nohud", "") + suffix):
            if os.path.isfile(cand):
                with open(cand) as fh:
                    d = json.load(fh)
                if d.get("mps") and any(v is not None for v in d["mps"]):
                    return [0.0 if v is None else float(v) for v in d["mps"]], cand
                if d.get("mph"):
                    return [0.0 if v is None else float(v) * MPH_TO_MPS
                            for v in d["mph"]], cand
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--lane-width", type=float, default=3.7,
                    help="known lane width in metres (US 3.7, EU 3.5)")
    ap.add_argument("--write", action="store_true", help="save models/calib/<stem>.json")
    ap.add_argument("--fallback-f", type=float, default=None,
                    help="f to assume when no speed log exists (default: 70 deg HFOV)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"{os.path.basename(args.video)}  {w}x{h} @ {fps:.2f} fps\n")

    vp = estimate_vanishing_point(cap)
    if vp is None:
        sys.exit("vanishing point estimation failed (too few lane lines)")
    print("1. vanishing point (Hough)")
    print(f"     u_vp = {vp['u_vp']:7.1f}   IQR {vp['u_iqr'][0]:.1f}..{vp['u_iqr'][1]:.1f}")
    print(f"     v_vp = {vp['v_vp']:7.1f}   IQR {vp['v_iqr'][0]:.1f}..{vp['v_iqr'][1]:.1f}"
          f"   (n={vp['n']})")

    hh = estimate_height(cap, (vp["u_vp"], vp["v_vp"]), args.lane_width)
    if hh is None:
        print("\n2. height: lane markings too weak; using 1.35 m dashcam prior.")
        print("     Lateral still uses the measured VP; range scale is the prior.")
        hh = {"cam_height_m": 1.35, "v_vp": vp["v_vp"], "rms_px": float("nan"), "n": 0}
    else:
        print(f"\n2. height from lane width ({args.lane_width:.2f} m)")
        print(f"     h    = {hh['cam_height_m']:7.3f} m")
        print(f"     v_vp = {hh['v_vp']:7.1f}  <-- independent check, "
              f"delta = {hh['v_vp'] - vp['v_vp']:+.1f} px")
        print(f"     fit rms {hh['rms_px']:.2f} px over n={hh['n']}")
        if abs(hh["v_vp"] - vp["v_vp"]) > 6.0:
            print("     WARNING: the two horizon estimates disagree by >6 px.")
            print("              Range will be unreliable. Check for a sloped road.")

    v_vp = 0.5 * (vp["v_vp"] + hh["v_vp"])
    cam_h = hh["cam_height_m"]

    speed, speed_path = load_speed(args.video, fps)
    prior = generic_calib(w, h)
    f_px, hf_info = (args.fallback_f if args.fallback_f else prior.f_px), None
    if speed is None:
        print("\n3. h*f: SKIPPED, no speed sidecar found.")
        print(f"     Using f = {f_px:.0f} px. Lateral offsets are still exact;")
        print("     forward range inherits this f's error linearly.")
    else:
        hf_info = estimate_hf(cap, v_vp, speed, fps=fps)
        if hf_info is None:
            print("\n3. h*f: fit failed; falling back.")
        else:
            f_px = hf_info["hf"] / cam_h
            print(f"\n3. h*f from optical flow vs {os.path.basename(speed_path)}")
            print(f"     h*f  = {hf_info['hf']:7.0f} px*m  "
                  f"95% CI {hf_info['ci95'][0]:.0f}..{hf_info['ci95'][1]:.0f}"
                  f"  (n={hf_info['n_obs']})")
            print(f"     f    = {f_px:7.0f} px")

    calib = GroundCalibration(
        source_width=w, source_height=h, f_px=float(f_px),
        u_vp=float(vp["u_vp"]), v_vp=float(v_vp), cam_height_m=float(cam_h),
        source=f"measured:{os.path.splitext(os.path.basename(args.video))[0]}",
    )
    print(f"\n=== RESULT ===")
    print(f"  f    = {calib.f_px:.0f} px   HFOV {calib.hfov_deg:.1f} deg")
    print(f"  u_vp = {calib.u_vp:.1f}   v_vp = {calib.v_vp:.1f}")
    print(f"  h    = {calib.cam_height_m:.3f} m   h*f = {calib.hf:.0f} px*m")
    print(f"  pitch {np.degrees(calib.pitch_rad):+.2f} deg   "
          f"yaw {np.degrees(calib.yaw_rad):+.2f} deg")
    print(f"\n  range sensitivity (1 px of bbox bottom jitter):")
    for z in (15, 30, 50, 70):
        print(f"     {z:3d} m -> +-{calib.range_sigma_m(z, 1.0):5.2f} m/px")
    print(f"  usable max range (1 px < 10% of range): "
          f"{calib.hf * 0.1:.0f} m -> capped at {calib.max_range_m:.0f} m")

    if args.write:
        out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "calib",
            os.path.splitext(os.path.basename(args.video))[0] + ".json")
        calib.to_json(out)
        print(f"\nwrote {out}")
    cap.release()


if __name__ == "__main__":
    main()
