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


def pair_gap_m(left, right, anchor_len=20):
    """Mean (right_x - left_x) on common visible anchors; None if unmatched."""
    xs_l, ys_l, zs_l, vis_l = parse_lane_components(left, anchor_len)
    xs_r, ys_r, zs_r, vis_r = parse_lane_components(right, anchor_len)
    common = vis_l & vis_r
    if int(common.sum()) >= 2:
        return float(np.mean(xs_r[common] - xs_l[common]))
    ml, mr = lane_mean_x(left, anchor_len), lane_mean_x(right, anchor_len)
    if ml is None or mr is None:
        return None
    return float(mr - ml)


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
        best, best_d = None, 1e9
        for lane in proposals:
            if exclude is not None and lane is exclude:
                continue
            mx = lane_mean_x(lane, anchor_len)
            if mx is None:
                continue
            d = abs(mx - target_x)
            if d < best_d and d <= self.match_x_m:
                best, best_d = lane, d
        return best

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
            jump = float(np.mean(np.abs(center[:, 0] - self._center[:, 0])))
            if jump > self.max_jump_m:
                self._center = center.copy()
                self._half_w = half_w.copy()
            else:
                a = self.alpha
                self._center = (1.0 - a) * self._center + a * center
                self._center[:, 1] = center[:, 1]
                self._half_w = (1.0 - a) * self._half_w + a * half_w

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

        common_vis = vis_curr & vis_next
        if common_vis.sum() >= 2:
            gap_m = float(np.mean(xs_next[common_vis] - xs_curr[common_vis]))
            if gap_m >= min_gap:
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


def extract_ego_corridor_3d(
    proposals,
    anchor_len=20,
    left_margin=None,
    right_margin=None,
    ego_left=None,
    ego_right=None,
):
    """
    Extracts matching (X, Y, Z) 3D coordinate arrays for Ego-Left and Ego-Right lanes.
    Defaults to symmetric corridor margins (P0).
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
        xs_l = xs_l + left_margin
    else:
        xs_l, ys_l, zs_l, vis_l = None, None, None, None

    if ego_right is not None:
        xs_r, ys_r, zs_r, vis_r = parse_lane_components(ego_right, anchor_len)
        xs_r = xs_r - right_margin
    else:
        xs_r, ys_r, zs_r, vis_r = None, None, None, None

    left_pts = []
    right_pts = []

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
            left_pts.append((xs_l[i], y_m, zs_l[i]))
            right_pts.append((xs_r[i], y_m, zs_r[i]))
        elif has_l:
            xl, zl = xs_l[i], zs_l[i]
            left_pts.append((xl, y_m, zl))
            right_pts.append((xl + STANDARD_LANE_WIDTH - (left_margin + right_margin), y_m, zl))
        elif has_r:
            xr, zr = xs_r[i], zs_r[i]
            left_pts.append((xr - STANDARD_LANE_WIDTH + (left_margin + right_margin), y_m, zr))
            right_pts.append((xr, y_m, zr))

    if len(left_pts) < 2 or len(right_pts) < 2:
        return None, None

    return np.array(left_pts), np.array(right_pts)


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
    if left_margin is None:
        left_margin = cfg.EGO_CORRIDOR_MARGIN_M
    if right_margin is None:
        right_margin = cfg.EGO_CORRIDOR_MARGIN_M

    use_smoothed_corridor = left_corridor_3d is not None and right_corridor_3d is not None
    if not use_smoothed_corridor and ego_left is None and ego_right is None:
        ego_left, ego_right = find_ego_lanes(proposals)

    if not use_smoothed_corridor and (ego_left is None or ego_right is None):
        return None

    model_w, model_h = img_size
    target_w, target_h = target_size
    scale_x = target_w / float(model_w)
    scale_y = target_h / float(model_h)

    # Project the same shared road-state corridor used by the BEV when it is
    # supplied.  It is already inset, so do not inset a second time in pixels.
    pts_l = decode_lane_pixels(
        left_corridor_3d if use_smoothed_corridor else ego_left,
        P_matrix,
        flat_ground=False,
    )
    pts_r = decode_lane_pixels(
        right_corridor_3d if use_smoothed_corridor else ego_right,
        P_matrix,
        flat_ground=False,
    )
    if len(pts_l) < 2 or len(pts_r) < 2:
        return None

    # Sort by v (top→bottom) and resample to equal count for a clean polygon
    def _sorted_xy(pts):
        arr = np.asarray(pts, dtype=np.float64)
        order = np.argsort(arr[:, 1])
        return arr[order]

    L = _sorted_xy(pts_l)
    R = _sorted_xy(pts_r)
    n = min(len(L), len(R), 16)
    # Resample along v
    v_min = max(float(L[0, 1]), float(R[0, 1]))
    v_max = min(float(L[-1, 1]), float(R[-1, 1]))
    if v_max - v_min < 5:
        return None
    vs = np.linspace(v_min, v_max, n)
    Lu = np.interp(vs, L[:, 1], L[:, 0])
    Ru = np.interp(vs, R[:, 1], R[:, 0])

    # Light inset toward centerline (fraction of measured pixel lane width)
    inset_frac_l = 0.0 if use_smoothed_corridor else float(left_margin) / float(STANDARD_LANE_WIDTH)
    inset_frac_r = 0.0 if use_smoothed_corridor else float(right_margin) / float(STANDARD_LANE_WIDTH)
    width = Ru - Lu
    # Guard collapsed / crossed samples
    width = np.maximum(width, 1.0)
    Lu_i = Lu + inset_frac_l * width
    Ru_i = Ru - inset_frac_r * width

    # Keep every sample — dropping OOB L/R unevenly created a thin skewed triangle
    def map_to_target(us, vs_):
        model_pts = np.column_stack((us, vs_))
        if model_to_target is not None:
            target_pts = np.asarray(model_to_target(model_pts), dtype=np.float64)
        else:
            target_pts = model_pts.copy()
            target_pts[:, 0] *= scale_x
            target_pts[:, 1] *= scale_y
        return [(int(round(u)), int(round(v))) for u, v in target_pts]

    left_pts = map_to_target(Lu_i, vs)
    right_pts = map_to_target(Ru_i, vs)
    if len(left_pts) < 2 or len(right_pts) < 2:
        return None

    poly_pts = left_pts + right_pts[::-1]
    return np.array(poly_pts, dtype=np.int32)
