import numpy as np
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels
from src.inference import lane_filter_config as cfg

STANDARD_LANE_WIDTH = 3.7  # Standard highway lane width in meters


def parse_lane_components(lane, anchor_len=20):
    """
    Unified parser for 3D lane representation.
    Supports both:
      1. Raw proposal vector of shape (86,) or (85,)
      2. Smoothed 3D point array of shape (N, 3) where columns are [X, Y, Z]
    Returns: (xs, ys, zs, vis) arrays of length anchor_len
    """
    if isinstance(lane, np.ndarray) and lane.ndim == 2 and lane.shape[1] == 3:
        xs = lane[:, 0].astype(np.float64)
        ys = lane[:, 1].astype(np.float64)
        zs = lane[:, 2].astype(np.float64)
        vis = np.ones(len(ys), dtype=bool)
    else:
        xs = lane[5:5 + anchor_len].astype(np.float64)
        zs = lane[5 + anchor_len:5 + 2 * anchor_len].astype(np.float64)
        vis = lane[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
        ys = ANCHOR_Y_STEPS[:len(xs)].astype(np.float64)
    return xs, ys, zs, vis


def lane_mean_x(lane, anchor_len=20, near_only=True):
    """Mean lateral X (m). Prefer near anchors — more stable for ego association."""
    xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
    if int(vis.sum()) < 2:
        return None
    if near_only and len(ys) >= 4:
        # Prefer Y <= 40 m when enough points (reduces far-curve bias)
        near = vis & (ys <= 40.0)
        if int(near.sum()) >= 2:
            return float(np.mean(xs[near]))
    return float(np.mean(xs[vis]))


def _aligned_gap_m(xs_l, ys_l, vis_l, xs_r, ys_r, vis_r):
    """Mean (right_x - left_x) on overlapping Y, even if sample counts differ."""
    if vis_l is None or vis_r is None:
        return None
    if int(np.sum(vis_l)) < 2 or int(np.sum(vis_r)) < 2:
        return None

    same_len = len(ys_l) == len(ys_r) == len(vis_l) == len(vis_r)
    if same_len and np.allclose(ys_l, ys_r, atol=1e-3):
        common = vis_l & vis_r
        if int(common.sum()) >= 2:
            return float(np.mean(xs_r[common] - xs_l[common]))

    y_l = ys_l[vis_l]
    x_l = xs_l[vis_l]
    y_r = ys_r[vis_r]
    x_r = xs_r[vis_r]
    order_l = np.argsort(y_l)
    order_r = np.argsort(y_r)
    y_l, x_l = y_l[order_l], x_l[order_l]
    y_r, x_r = y_r[order_r], x_r[order_r]
    y_min = max(float(y_l[0]), float(y_r[0]))
    y_max = min(float(y_l[-1]), float(y_r[-1]))
    if y_max - y_min < 5.0:
        return None
    y_c = np.linspace(y_min, y_max, 8)
    return float(np.mean(np.interp(y_c, y_r, x_r) - np.interp(y_c, y_l, x_l)))


def pair_gap_m(left, right, anchor_len=20):
    """Mean (right_x - left_x) on overlapping Y; None if unmatched."""
    xs_l, ys_l, zs_l, vis_l = parse_lane_components(left, anchor_len)
    xs_r, ys_r, zs_r, vis_r = parse_lane_components(right, anchor_len)
    gap = _aligned_gap_m(xs_l, ys_l, vis_l, xs_r, ys_r, vis_r)
    if gap is not None:
        return gap
    ml, mr = lane_mean_x(left, anchor_len), lane_mean_x(right, anchor_len)
    if ml is None or mr is None:
        return None
    return float(mr - ml)


def lane_points_3d(lane, anchor_len=20):
    """Visible 3D points as (N, 3) [X, Y, Z], or None if too short."""
    xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
    if int(vis.sum()) < 2:
        return None
    return np.column_stack((xs[vis], ys[vis], zs[vis])).astype(np.float64)


def clip_lane_to_max_y(lane, max_y_m, anchor_len=20, min_points=3):
    """Keep visible samples with Y <= max_y_m. Returns (N, 3) or None."""
    pts = lane_points_3d(lane, anchor_len)
    if pts is None:
        return None
    pts = pts[pts[:, 1] <= float(max_y_m) + 1e-6]
    if len(pts) < int(min_points):
        return None
    return pts.astype(np.float32)


def match_lane_by_x(target_x, proposals, match_x_m, anchor_len=20, exclude=None):
    """Nearest proposal whose mean X is within match_x_m of target_x."""
    if target_x is None or proposals is None:
        return None
    best, best_d = None, 1e9
    for lane in proposals:
        if exclude is not None and lane is exclude:
            continue
        mx = lane_mean_x(lane, anchor_len)
        if mx is None:
            continue
        d = abs(mx - float(target_x))
        if d < best_d and d <= float(match_x_m):
            best, best_d = lane, d
    return best


def select_onesided_ego_lane(
    proposals,
    left_x,
    right_x,
    match_x_m=None,
    width_min=None,
    width_max=None,
    anchor_len=20,
):
    """
    Pick the remaining ego boundary when the other side is missing.

    Returns (side, lane) where side is 'left' or 'right' (the *visible* side),
    or (None, None) if both/neither sides rematch as a width-valid pair.
    """
    if proposals is None or len(proposals) == 0:
        return None, None
    if left_x is None and right_x is None:
        return None, None

    match_x_m = cfg.ONESIDED_MATCH_X_M if match_x_m is None else match_x_m
    width_min = cfg.EGO_LANE_WIDTH_MIN_M if width_min is None else width_min
    width_max = cfg.EGO_LANE_WIDTH_MAX_M if width_max is None else width_max

    left_match = match_lane_by_x(left_x, proposals, match_x_m, anchor_len) if left_x is not None else None
    right_match = (
        match_lane_by_x(right_x, proposals, match_x_m, anchor_len, exclude=left_match)
        if right_x is not None else None
    )

    has_l = left_match is not None
    has_r = right_match is not None
    if has_l and not has_r:
        return "left", left_match
    if has_r and not has_l:
        return "right", right_match
    if not has_l and not has_r:
        return None, None

    gap = pair_gap_m(left_match, right_match, anchor_len)
    if gap is not None and width_min <= gap <= width_max:
        return None, None

    # Both rematched but the span is not one lane (usually the "missing" side
    # snapped onto an adjacent marking). Keep the closer historical match.
    mx_l = lane_mean_x(left_match, anchor_len)
    mx_r = lane_mean_x(right_match, anchor_len)
    d_l = abs(mx_l - float(left_x)) if mx_l is not None else 1e9
    d_r = abs(mx_r - float(right_x)) if mx_r is not None else 1e9
    if d_l + 0.3 < d_r:
        return "left", left_match
    if d_r + 0.3 < d_l:
        return "right", right_match
    if mx_l is not None and mx_r is not None and abs(mx_l) <= abs(mx_r):
        return "left", left_match
    return "right", right_match


def reconstruct_opposite_boundary(
    visible_lane,
    visible_side,
    width_m,
    max_y_m=None,
    anchor_len=20,
):
    """
    Build the missing ego boundary from a live side + locked lane width.

    visible_side='left'  → reconstructed is to the right (X + W)
    visible_side='right' → reconstructed is to the left  (X - W)
    Only Y <= max_y_m is emitted so far-range paint is not invented.
    """
    max_y_m = cfg.ONESIDED_MAX_Y_M if max_y_m is None else max_y_m
    pts = clip_lane_to_max_y(visible_lane, max_y_m, anchor_len=anchor_len)
    if pts is None:
        return None
    out = pts.copy()
    w = float(width_m)
    if visible_side == "left":
        out[:, 0] = pts[:, 0] + w
    elif visible_side == "right":
        out[:, 0] = pts[:, 0] - w
    else:
        return None
    return out.astype(np.float32)


def _legacy_ego_lanes(proposals, anchor_len=20):
    """Old sign-of-X heuristic (fallback when no width-valid pair)."""
    left_lanes = []
    right_lanes = []
    for lane in proposals:
        mx = lane_mean_x(lane, anchor_len)
        if mx is None:
            continue
        if mx <= 0:
            left_lanes.append((abs(mx), lane))
        else:
            right_lanes.append((mx, lane))
    ego_left = sorted(left_lanes, key=lambda item: item[0])[0][1] if left_lanes else None
    ego_right = sorted(right_lanes, key=lambda item: item[0])[0][1] if right_lanes else None
    return ego_left, ego_right


def find_ego_lanes(
    proposals,
    anchor_len=20,
    width_min=None,
    width_max=None,
    width_target=None,
):
    """
    P0 ego pair: pick left/right boundaries that form ONE real lane.

    Prefer pairs that:
      1. Bracket ego center (left_x < 0 < right_x), and
      2. Have gap in [width_min, width_max] (~one lane), closest to width_target,
      3. Have corridor center closest to 0.

    Falls back to legacy closest-left / closest-right if no width-valid pair.
    """
    if proposals is None or len(proposals) == 0:
        return None, None

    width_min = cfg.EGO_LANE_WIDTH_MIN_M if width_min is None else width_min
    width_max = cfg.EGO_LANE_WIDTH_MAX_M if width_max is None else width_max
    width_target = cfg.EGO_LANE_WIDTH_TARGET_M if width_target is None else width_target

    lanes = []
    for lane in proposals:
        mx = lane_mean_x(lane, anchor_len)
        if mx is None:
            continue
        lanes.append((mx, lane))
    if len(lanes) < 2:
        return _legacy_ego_lanes(proposals, anchor_len)

    lanes.sort(key=lambda t: t[0])
    best = None  # (score, left, right)
    for i in range(len(lanes)):
        for j in range(i + 1, len(lanes)):
            ml, left = lanes[i]
            mr, right = lanes[j]
            # Must bracket ego (allow slight straddle tolerance)
            if not (ml < 0.35 and mr > -0.35 and ml < mr):
                continue
            gap = pair_gap_m(left, right, anchor_len)
            if gap is None or gap < width_min or gap > width_max:
                continue
            center = 0.5 * (ml + mr)
            # Lower is better: prefer target width + centered corridor
            score = abs(gap - width_target) + 0.75 * abs(center)
            if best is None or score < best[0]:
                best = (score, left, right)

    if best is not None:
        return best[1], best[2]
    return _legacy_ego_lanes(proposals, anchor_len)


class EgoLanePairTracker:
    """
    Temporal hold for ego left/right during lane-change / flicker.

    Accepts a fresh width-valid pair immediately; otherwise rematches the
    last good pair into the current proposal set for up to hold_frames.
    """

    def __init__(
        self,
        hold_frames=None,
        match_x_m=None,
        width_min=None,
        width_max=None,
    ):
        self.hold_frames = cfg.EGO_PAIR_HOLD_FRAMES if hold_frames is None else hold_frames
        self.match_x_m = cfg.EGO_PAIR_MATCH_X_M if match_x_m is None else match_x_m
        self.width_min = cfg.EGO_LANE_WIDTH_MIN_M if width_min is None else width_min
        self.width_max = cfg.EGO_LANE_WIDTH_MAX_M if width_max is None else width_max
        self._left = None
        self._right = None
        self._left_x = None
        self._right_x = None
        self._miss = 0
        self.last_meta = {
            "source": "none",
            "gap_m": None,
            "center_m": None,
            "held": False,
        }

    def reset(self):
        self._left = self._right = None
        self._left_x = self._right_x = None
        self._miss = 0
        self.last_meta = {"source": "none", "gap_m": None, "center_m": None, "held": False}

    def _match_lane(self, target_x, proposals, anchor_len=20, exclude=None):
        return match_lane_by_x(target_x, proposals, self.match_x_m, anchor_len, exclude=exclude)

    def _set_pair(self, left, right, source, held, anchor_len=20):
        self._left, self._right = left, right
        self._left_x = lane_mean_x(left, anchor_len)
        self._right_x = lane_mean_x(right, anchor_len)
        gap = pair_gap_m(left, right, anchor_len)
        center = None
        if self._left_x is not None and self._right_x is not None:
            center = 0.5 * (self._left_x + self._right_x)
        self.last_meta = {
            "source": source,
            "gap_m": gap,
            "center_m": center,
            "held": held,
            "left_x": self._left_x,
            "right_x": self._right_x,
        }
        if not held:
            self._miss = 0
        return left, right

    def update(self, proposals, anchor_len=20):
        if proposals is None or len(proposals) == 0:
            self._miss += 1
            if self._left is not None and self._miss <= self.hold_frames:
                self.last_meta["held"] = True
                self.last_meta["source"] = "hold_empty"
                return self._left, self._right
            self.reset()
            return None, None

        cand_l, cand_r = find_ego_lanes(
            proposals,
            anchor_len=anchor_len,
            width_min=self.width_min,
            width_max=self.width_max,
        )
        if cand_l is not None and cand_r is not None:
            gap = pair_gap_m(cand_l, cand_r, anchor_len)
            if gap is not None and self.width_min <= gap <= self.width_max:
                return self._set_pair(cand_l, cand_r, "width_pair", False, anchor_len)

        # Rematch last good lateral positions into this frame's proposals
        if self._left_x is not None and self._right_x is not None:
            rem_l = self._match_lane(self._left_x, proposals, anchor_len)
            rem_r = self._match_lane(self._right_x, proposals, anchor_len, exclude=rem_l)
            if rem_l is not None and rem_r is not None:
                gap = pair_gap_m(rem_l, rem_r, anchor_len)
                # Keep held corridor only if rematch is still ~one lane (or slightly wide)
                if gap is not None and self.width_min * 0.85 <= gap <= self.width_max * 1.15:
                    self._miss += 1
                    if self._miss <= self.hold_frames:
                        return self._set_pair(rem_l, rem_r, "hold_rematch", True, anchor_len)

            # Rematch failed / too wide: keep previous pair objects for a few frames
            self._miss += 1
            if self._miss <= self.hold_frames:
                self.last_meta["held"] = True
                self.last_meta["source"] = "hold_stale"
                return self._left, self._right

        # Soft fallback: legacy pair only when we have no held corridor yet
        if cand_l is not None and cand_r is not None and self._left is None:
            gap = pair_gap_m(cand_l, cand_r, anchor_len)
            # Still reject obviously multi-lane spans even as cold start
            if gap is not None and gap <= self.width_max * 1.25:
                return self._set_pair(cand_l, cand_r, "legacy_fallback", False, anchor_len)

        self.reset()
        return None, None


def find_outer_lanes(proposals, anchor_len=20, max_abs_x=9.0):
    """
    Outermost detected left / right lanes for dynamic BEV road edges (P1).
    Returns (outer_left, outer_right) — may be None independently.
    """
    if proposals is None or len(proposals) == 0:
        return None, None
    left_best = None
    right_best = None
    for lane in proposals:
        mx = lane_mean_x(lane, anchor_len)
        if mx is None or abs(mx) > max_abs_x:
            continue
        if mx < 0:
            if left_best is None or mx < left_best[0]:
                left_best = (mx, lane)
        else:
            if right_best is None or mx > right_best[0]:
                right_best = (mx, lane)
    return (
        left_best[1] if left_best is not None else None,
        right_best[1] if right_best is not None else None,
    )


class CorridorEMA:
    """
    Temporally smooth ego corridor WITHOUT introducing shear.

    Smooth centerline + width jointly (not left/right independently).
    """

    def __init__(self, alpha=None, max_jump_m=None):
        self.alpha = cfg.CORRIDOR_EMA_ALPHA if alpha is None else alpha
        self.max_jump_m = cfg.CORRIDOR_EMA_MAX_JUMP_M if max_jump_m is None else max_jump_m
        self._center = None  # (N, 3) midpoints
        self._half_w = None  # (N,) half-widths

    def reset(self):
        self._center = None
        self._half_w = None

    @staticmethod
    def _resample(pts, n=20):
        if pts is None or len(pts) < 2:
            return None
        pts = np.asarray(pts, dtype=np.float64)
        order = np.argsort(pts[:, 1])
        pts = pts[order]
        y_new = np.linspace(float(pts[0, 1]), float(pts[-1, 1]), n)
        x_new = np.interp(y_new, pts[:, 1], pts[:, 0])
        z_new = np.interp(y_new, pts[:, 1], pts[:, 2])
        return np.stack([x_new, y_new, z_new], axis=1)

    def update(self, left_3d, right_3d):
        left = self._resample(left_3d)
        right = self._resample(right_3d)
        if left is None or right is None:
            self.reset()
            return left_3d, right_3d

        n = min(len(left), len(right))
        left, right = left[:n], right[:n]
        center = 0.5 * (left + right)
        half_w = 0.5 * (right[:, 0] - left[:, 0])

        if self._center is None:
            self._center = center.copy()
            self._half_w = half_w.copy()
        else:
            y = center[:, 1]
            prev_cx = np.interp(y, self._center[:, 1], self._center[:, 0])
            prev_cz = np.interp(y, self._center[:, 1], self._center[:, 2])
            prev_hw = np.interp(y, self._center[:, 1], self._half_w)
            jump = float(np.mean(np.abs(center[:, 0] - prev_cx)))
            if jump > self.max_jump_m:
                self._center = center.copy()
                self._half_w = half_w.copy()
            else:
                a = self.alpha
                self._center = center.copy()
                self._center[:, 0] = (1.0 - a) * prev_cx + a * center[:, 0]
                self._center[:, 2] = (1.0 - a) * prev_cz + a * center[:, 2]
                self._half_w = (1.0 - a) * prev_hw + a * half_w

        left_out = self._center.copy()
        right_out = self._center.copy()
        left_out[:, 0] = self._center[:, 0] - self._half_w
        right_out[:, 0] = self._center[:, 0] + self._half_w
        return left_out, right_out


def fill_missing_lane_gaps(
    proposals,
    anchor_len=20,
    standard_lane_width=3.7,
    min_gap=None,
    enabled=None,
    min_score=None,
):
    """
    Optionally interpolate missing intermediate lane lines when adjacent gaps are large.

    Disabled by default (see lane_filter_config.ENABLE_FILL_MISSING_LANES) because it
    invents false positives when detections are noisy.
    """
    if enabled is None:
        enabled = cfg.ENABLE_FILL_MISSING_LANES
    if not enabled or proposals is None or len(proposals) == 0:
        return proposals

    min_gap = cfg.FILL_MIN_GAP_M if min_gap is None else min_gap
    min_score = cfg.FILL_MIN_SCORE if min_score is None else min_score

    valid_proposals = []
    for lane in proposals:
        xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
        if vis.sum() < 2:
            continue
        if isinstance(lane, np.ndarray) and lane.ndim == 1 and lane.shape[0] > 1:
            if float(lane[1]) < min_score:
                continue
        mean_x = float(np.mean(xs[vis]))
        valid_proposals.append((mean_x, lane))

    if len(valid_proposals) < 2:
        return proposals

    valid_proposals = sorted(valid_proposals, key=lambda item: item[0])
    augmented_proposals = [item[1] for item in valid_proposals]

    i = 0
    while i < len(augmented_proposals) - 1:
        lane_curr = augmented_proposals[i]
        lane_next = augmented_proposals[i + 1]

        xs_curr, ys_curr, zs_curr, vis_curr = parse_lane_components(lane_curr, anchor_len)
        xs_next, ys_next, zs_next, vis_next = parse_lane_components(lane_next, anchor_len)

        gap_m = _aligned_gap_m(xs_curr, ys_curr, vis_curr, xs_next, ys_next, vis_next)
        if gap_m is not None and gap_m >= min_gap:
            num_missing = int(round(gap_m / standard_lane_width)) - 1
            for step in range(1, num_missing + 1):
                shift_offset = step * standard_lane_width
                if isinstance(lane_curr, np.ndarray) and lane_curr.ndim == 2 and lane_curr.shape[1] == 3:
                    synth_lane = lane_curr.copy()
                    synth_lane[:, 0] += shift_offset
                else:
                    synth_lane = lane_curr.copy()
                    synth_lane[5:5 + anchor_len] = xs_curr + shift_offset
                augmented_proposals.insert(i + step, synth_lane)
            i += num_missing
        i += 1

    return augmented_proposals


def _clamp_ego_gap(xl, xr):
    """Keep corridor at one-lane width. Shrink oversized pairs around the center."""
    gap = float(xr) - float(xl)
    if gap <= 0:
        return xl, xr
    target = float(getattr(cfg, "CORRIDOR_WIDTH_CLAMP_M", cfg.EGO_LANE_WIDTH_TARGET_M))
    # Fixed-width mode: always use the defined paint width (EKF-safe).
    if bool(getattr(cfg, "CORRIDOR_FORCE_FIXED_WIDTH", False)):
        center = 0.5 * (float(xl) + float(xr))
        half = 0.5 * target
        return center - half, center + half
    width_max = float(getattr(cfg, "CORRIDOR_WIDTH_MAX_M", 3.9))
    if gap <= width_max:
        return xl, xr
    center = 0.5 * (float(xl) + float(xr))
    half = 0.5 * target
    return center - half, center + half


def force_corridor_fixed_width(left_3d, right_3d, width_m=None, margin_m=None):
    """Rebuild corridor edges from centerline at a fixed paint width, then inset.

    Used so EKF / PREDICTED fallback cannot inflate the drivable fill beyond the
    defined lane width. Returns (left, right) or (None, None).
    """
    if left_3d is None or right_3d is None:
        return None, None
    left = np.asarray(left_3d, dtype=np.float64)
    right = np.asarray(right_3d, dtype=np.float64)
    n = min(len(left), len(right))
    if n < 2:
        return None, None
    left, right = left[:n].copy(), right[:n].copy()

    paint_w = float(
        cfg.CORRIDOR_WIDTH_CLAMP_M if width_m is None else width_m
    )
    if not np.isfinite(paint_w) or paint_w < 1.0:
        paint_w = float(cfg.EGO_LANE_WIDTH_TARGET_M)
    margin = float(cfg.EGO_CORRIDOR_MARGIN_M if margin_m is None else margin_m)
    fill_w = max(0.5, paint_w - 2.0 * margin)
    half = 0.5 * fill_w

    center_x = 0.5 * (left[:, 0] + right[:, 0])
    left[:, 0] = center_x - half
    right[:, 0] = center_x + half
    return left, right


def extract_ego_corridor_3d(
    proposals,
    anchor_len=20,
    left_margin=None,
    right_margin=None,
    ego_left=None,
    ego_right=None,
    lane_width_m=None,
):
    """
    Extracts matching (X, Y, Z) 3D coordinate arrays for Ego-Left and Ego-Right lanes.
    Clamp wide pairs to one lane, then inset by corridor margins.
    """
    if left_margin is None:
        left_margin = cfg.EGO_CORRIDOR_MARGIN_M
    if right_margin is None:
        right_margin = cfg.EGO_CORRIDOR_MARGIN_M

    if ego_left is None and ego_right is None:
        ego_left, ego_right = find_ego_lanes(proposals, anchor_len)

    if ego_left is None and ego_right is None:
        return None, None

    if ego_left is not None:
        xs_l, ys_l, zs_l, vis_l = parse_lane_components(ego_left, anchor_len)
    else:
        xs_l, ys_l, zs_l, vis_l = None, None, None, None

    if ego_right is not None:
        xs_r, ys_r, zs_r, vis_r = parse_lane_components(ego_right, anchor_len)
    else:
        xs_r, ys_r, zs_r, vis_r = None, None, None, None

    left_pts = []
    right_pts = []
    width_m = float(STANDARD_LANE_WIDTH if lane_width_m is None else lane_width_m)

    num_steps = min(
        anchor_len,
        len(ys_l) if ys_l is not None else anchor_len,
        len(ys_r) if ys_r is not None else anchor_len,
    )
    y_steps = ys_l if ys_l is not None else (ys_r if ys_r is not None else ANCHOR_Y_STEPS)

    for i in range(min(num_steps, len(y_steps))):
        y_m = y_steps[i]

        has_l = vis_l[i] if vis_l is not None else False
        has_r = vis_r[i] if vis_r is not None else False

        if has_l and has_r:
            xl, xr = _clamp_ego_gap(xs_l[i], xs_r[i])
        elif has_l:
            xl = xs_l[i]
            xr = xl + width_m
            xl, xr = _clamp_ego_gap(xl, xr)
        elif has_r:
            xr = xs_r[i]
            xl = xr - width_m
            xl, xr = _clamp_ego_gap(xl, xr)
        else:
            continue

        zl = zs_l[i] if has_l else (zs_r[i] if has_r else 0.0)
        zr = zs_r[i] if has_r else zl
        left_pts.append((xl + left_margin, y_m, zl))
        right_pts.append((xr - right_margin, y_m, zr))

    if len(left_pts) < 2 or len(right_pts) < 2:
        return None, None

    return np.array(left_pts), np.array(right_pts)


def _clip_corridor_3d_near(lane_3d, y_start):
    """Drop 3D samples on/under the ego hood so the fill starts on the road."""
    if lane_3d is None:
        return None
    arr = np.asarray(lane_3d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return lane_3d
    keep = arr[:, 1] >= float(y_start)
    if int(keep.sum()) < 2:
        return lane_3d
    return arr[keep]


def get_ego_corridor_sides_2d(
    proposals,
    P_matrix,
    img_size=(480, 360),
    target_size=(1080, 720),
    left_margin=None,
    right_margin=None,
    ego_left=None,
    ego_right=None,
    model_to_target=None,
    left_corridor_3d=None,
    right_corridor_3d=None,
):
    """
    Ego-corridor left/right polylines in image pixels (top → bottom).

    Geometry: 3D ego pair + lateral margin inset (EGO_CORRIDOR_MARGIN_M).
    Near clip: skip Y < CORRIDOR_Y_START_M and image rows on the bonnet
    (CORRIDOR_IMAGE_HOOD_FRAC from the bottom).
    """
    if left_margin is None:
        left_margin = cfg.EGO_CORRIDOR_MARGIN_M
    if right_margin is None:
        right_margin = cfg.EGO_CORRIDOR_MARGIN_M

    y_start = float(getattr(cfg, "CORRIDOR_Y_START_M", 4.0))
    hood_frac = float(getattr(cfg, "CORRIDOR_IMAGE_HOOD_FRAC", 0.14))

    use_smoothed_corridor = left_corridor_3d is not None and right_corridor_3d is not None
    if not use_smoothed_corridor and ego_left is None and ego_right is None:
        ego_left, ego_right = find_ego_lanes(proposals)

    if not use_smoothed_corridor and (ego_left is None or ego_right is None):
        return None

    model_w, model_h = img_size
    target_w, target_h = target_size
    scale_x = target_w / float(model_w)
    scale_y = target_h / float(model_h)

    src_l = left_corridor_3d if use_smoothed_corridor else ego_left
    src_r = right_corridor_3d if use_smoothed_corridor else ego_right
    if use_smoothed_corridor:
        src_l = _clip_corridor_3d_near(src_l, y_start)
        src_r = _clip_corridor_3d_near(src_r, y_start)

    pts_l = decode_lane_pixels(src_l, P_matrix, flat_ground=False)
    pts_r = decode_lane_pixels(src_r, P_matrix, flat_ground=False)
    if len(pts_l) < 2 or len(pts_r) < 2:
        return None

    def _sorted_xy(pts):
        arr = np.asarray(pts, dtype=np.float64)
        order = np.argsort(arr[:, 1])
        return arr[order]

    L = _sorted_xy(pts_l)
    R = _sorted_xy(pts_r)
    n = min(len(L), len(R), 16)
    v_min = max(float(L[0, 1]), float(R[0, 1]))
    v_max = min(float(L[-1, 1]), float(R[-1, 1]))
    if v_max - v_min < 5:
        return None
    vs = np.linspace(v_min, v_max, n)
    Lu = np.interp(vs, L[:, 1], L[:, 0])
    Ru = np.interp(vs, R[:, 1], R[:, 0])

    inset_frac_l = 0.0 if use_smoothed_corridor else float(left_margin) / float(STANDARD_LANE_WIDTH)
    inset_frac_r = 0.0 if use_smoothed_corridor else float(right_margin) / float(STANDARD_LANE_WIDTH)
    width = np.maximum(Ru - Lu, 1.0)
    Lu_i = Lu + inset_frac_l * width
    Ru_i = Ru - inset_frac_r * width

    def map_to_target(us, vs_):
        model_pts = np.column_stack((us, vs_))
        if model_to_target is not None:
            target_pts = np.asarray(model_to_target(model_pts), dtype=np.float64)
        else:
            target_pts = model_pts.copy()
            target_pts[:, 0] *= scale_x
            target_pts[:, 1] *= scale_y
        return target_pts

    left_xy = map_to_target(Lu_i, vs)
    right_xy = map_to_target(Ru_i, vs)
    v_hood = float(target_h) * (1.0 - hood_frac)
    keep = (left_xy[:, 1] <= v_hood) & (right_xy[:, 1] <= v_hood)
    if int(keep.sum()) < 2:
        return None
    left_xy = left_xy[keep]
    right_xy = right_xy[keep]
    left_pts = [(int(round(u)), int(round(v))) for u, v in left_xy]
    right_pts = [(int(round(u)), int(round(v))) for u, v in right_xy]
    if len(left_pts) < 2 or len(right_pts) < 2:
        return None
    return left_pts, right_pts


def get_ego_corridor_2d_pixels(
    proposals,
    P_matrix,
    img_size=(480, 360),
    target_size=(1080, 720),
    left_margin=None,
    right_margin=None,
    ego_left=None,
    ego_right=None,
    model_to_target=None,
    left_corridor_3d=None,
    right_corridor_3d=None,
):
    """
    Build drivable polygon in IMAGE SPACE from the same projected ego polylines
    used for lane drawing — guarantees the fill sits between the overlay lines.

    Inset is a fraction of lane width in pixels (not a re-project with different Z).
    """
    sides = get_ego_corridor_sides_2d(
        proposals,
        P_matrix,
        img_size=img_size,
        target_size=target_size,
        left_margin=left_margin,
        right_margin=right_margin,
        ego_left=ego_left,
        ego_right=ego_right,
        model_to_target=model_to_target,
        left_corridor_3d=left_corridor_3d,
        right_corridor_3d=right_corridor_3d,
    )
    if sides is None:
        return None
    left_pts, right_pts = sides
    return np.array(left_pts + right_pts[::-1], dtype=np.int32)
