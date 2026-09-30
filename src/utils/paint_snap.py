"""Pull a projected lane polyline onto the white or yellow paint.

The 3D lane is in the OpenLane camera, so on this dashcam the projected line
crosses the real marking. Each sample is moved sideways to the nearest thin
paint run on its own image row. Gaps in a dashed line keep the shift of the
nearest sample that did find paint.
"""

from __future__ import annotations

import cv2
import numpy as np


def lane_paint_mask(frame: np.ndarray, calib=None) -> np.ndarray:
    """1 on thin lane paint. Wide bright things (cars, trees, signs) are dropped."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Top-hat keeps structures thinner than the kernel. A lane stripe is thin.
    # A car, a tree, or a building is not.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    white = (hat > 16) & (gray > 135) & (hsv[:, :, 1] < 110)
    yellow = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 120)
        & (hat > 10)
    )
    mask = (white | yellow).astype(np.uint8)
    if calib is not None:
        top = int(min(h - 2, max(0, float(calib.v_vp) + 6)))
    else:
        top = int(0.48 * h)
    mask[:top, :] = 0
    mask[int(0.94 * h) :, :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        thickness = area / float(max(bw, bh, 1))
        # Lane paint is a few pixels thick. A car or a tree is not.
        if thickness > 14.0:
            mask[labels == i] = 0
    return mask


def snap_polyline_to_paint(pts, frame, calib=None, mask=None, window_m=2.2):
    """Return the polyline with each point shifted onto nearby paint.

    `calib` is the dashcam ground calibration. It sets the search width in
    metres so a near-field line can reach the paint without jumping a whole
    lane. Without it, the search widens toward the bottom of the frame.
    """
    if pts is None or len(pts) < 2:
        return list(pts) if pts is not None else []
    h, w = frame.shape[:2]
    if mask is None:
        mask = lane_paint_mask(frame, calib)

    src = [(float(u), float(v)) for u, v in pts]
    order = sorted(range(len(src)), key=lambda i: -src[i][1])
    # Shift is stored in metres when the dashcam calibration is known, so a
    # correction measured on a far dash still lands on the near paint.
    dx_m = [0.0] * len(src)
    du_px = [0.0] * len(src)
    hit = [False] * len(src)
    prev_m = None
    for i in order:
        u, v = src[i]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < h and 0 <= ui < w):
            continue
        if vi < int(0.40 * h) or vi > int(0.94 * h):
            continue
        half = _half_window_px(calib, vi, h, window_m)
        found = _nearest_paint_u(mask, ui, vi, half)
        if found is None:
            continue
        shift_px = float(found) - float(ui)
        shift_m = _px_to_m(calib, vi, shift_px)
        if prev_m is not None and abs(shift_m - prev_m) > 0.9:
            continue
        dx_m[i] = shift_m
        du_px[i] = shift_px
        hit[i] = True
        prev_m = shift_m

    if not any(hit):
        return [(int(round(u)), int(round(v))) for u, v in src]

    known = [i for i in range(len(src)) if hit[i]]
    if len(known) >= 3:
        vals = [dx_m[i] for i in known]
        for k, i in enumerate(known):
            lo, hi = max(0, k - 1), min(len(vals), k + 2)
            dx_m[i] = float(np.median(vals[lo:hi]))
    for i in range(len(src)):
        if hit[i]:
            continue
        nearest = min(known, key=lambda k: abs(src[k][1] - src[i][1]))
        dx_m[i] = dx_m[nearest]
        du_px[i] = _m_to_px(calib, src[i][1], dx_m[i], fallback_px=du_px[nearest])

    out = []
    for i, (u, v) in enumerate(src):
        uu = u + du_px[i]
        if 0.0 <= uu < w and 0.0 <= v < h:
            out.append((int(round(uu)), int(round(v))))
    return out


def snap_hit_fraction(pts, frame, calib=None, mask=None, window_m=2.2) -> float:
    """Share of polyline samples that land on paint after the snap search."""
    if not pts:
        return 0.0
    h, _w = frame.shape[:2]
    if mask is None:
        mask = lane_paint_mask(frame, calib)
    hits = 0
    used = 0
    for u, v in pts:
        vi = int(round(v))
        ui = int(round(u))
        if not (int(0.40 * h) <= vi <= int(0.94 * h)):
            continue
        used += 1
        half = _half_window_px(calib, vi, h, window_m)
        if _nearest_paint_u(mask, ui, vi, half) is not None:
            hits += 1
    if used == 0:
        return 0.0
    return hits / float(used)


def _px_to_m(calib, v, shift_px):
    if calib is None:
        return float(shift_px)
    dv = float(v) - float(calib.v_vp)
    if dv < 8.0 or calib.f_px < 1.0:
        return float(shift_px)
    y = float(calib.hf) / dv
    return float(shift_px) * y / float(calib.f_px)


def _m_to_px(calib, v, shift_m, fallback_px=0.0):
    if calib is None:
        return float(fallback_px)
    dv = float(v) - float(calib.v_vp)
    if dv < 8.0 or calib.f_px < 1.0:
        return float(fallback_px)
    y = float(calib.hf) / dv
    return float(shift_m) * float(calib.f_px) / y


def _half_window_px(calib, v, h, window_m):
    if calib is not None:
        dv = float(v) - float(calib.v_vp)
        if dv > 8.0:
            y = float(calib.hf) / dv
            return int(max(10.0, min(140.0, float(calib.f_px) * float(window_m) / max(y, 1.0))))
    t = (float(v) - 0.40 * h) / max(1.0, 0.54 * h)
    t = float(np.clip(t, 0.0, 1.0))
    return int(12 + t * 80)


def _nearest_paint_u(mask, u, v, half):
    h, w = mask.shape[:2]
    if not (0 <= v < h):
        return None
    u0 = max(0, int(u) - int(half))
    u1 = min(w - 1, int(u) + int(half))
    if u1 - u0 < 3:
        return None
    cols = np.flatnonzero(mask[v, u0 : u1 + 1] > 0)
    if cols.size < 2:
        return None
    cols = cols + u0
    groups = []
    cur = [int(cols[0])]
    for c in cols[1:]:
        c = int(c)
        if c - cur[-1] <= 6:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    max_run = max(18, int(0.45 * (u1 - u0)))
    best = None
    best_dist = 1e9
    for g in groups:
        if len(g) < 2 or len(g) > max_run:
            continue
        # A run that touches the search edge is usually a car or a shadow, not a stripe.
        if g[0] <= u0 + 1 or g[-1] >= u1 - 1:
            continue
        center = 0.5 * (g[0] + g[-1])
        dist = abs(center - float(u))
        if dist < best_dist:
            best_dist = dist
            best = center
    return best
