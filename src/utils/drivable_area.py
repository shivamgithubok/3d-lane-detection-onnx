import numpy as np
import cv2
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels

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


def find_ego_lanes(proposals, anchor_len=20):
    """
    Identifies the Ego-Left lane (closest lane to left of vehicle center, X <= 0)
    and Ego-Right lane (closest lane to right of vehicle center, X >= 0).
    """
    if proposals is None or len(proposals) == 0:
        return None, None

    left_lanes = []
    right_lanes = []

    for lane in proposals:
        xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
        if vis.sum() < 2:
            continue
        valid_xs = xs[vis]
        mean_x = float(np.mean(valid_xs))

        if mean_x <= 0:
            left_lanes.append((abs(mean_x), lane))
        else:
            right_lanes.append((mean_x, lane))

    ego_left = sorted(left_lanes, key=lambda item: item[0])[0][1] if left_lanes else None
    ego_right = sorted(right_lanes, key=lambda item: item[0])[0][1] if right_lanes else None

    return ego_left, ego_right


def fill_missing_lane_gaps(proposals, anchor_len=20, standard_lane_width=3.7, min_gap=5.0):
    """
    Inspects detected 3D lane proposals and automatically interpolates missing intermediate lane lines
    when a gap between adjacent detected lanes is >= min_gap (5.0m).
    """
    if proposals is None or len(proposals) == 0:
        return proposals

    valid_proposals = []
    for lane in proposals:
        xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
        if vis.sum() >= 2:
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


def extract_ego_corridor_3d(proposals, anchor_len=20, left_margin=0.70, right_margin=1.20):
    """
    Extracts matching (X, Y, Z) 3D coordinate arrays for Ego-Left and Ego-Right lanes,
    applying left_margin=0.70m and right_margin=1.20m.
    """
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

    num_steps = min(anchor_len, len(ys_l) if ys_l is not None else anchor_len, len(ys_r) if ys_r is not None else anchor_len)
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


def get_ego_corridor_2d_pixels(proposals, P_matrix, img_size=(480, 360), target_size=(1080, 720), left_margin=0.70, right_margin=1.20):
    """
    Projects Ego-Left and Ego-Right 3D lane lines directly onto 2D camera pixels,
    applying left_margin and right_margin to sit cleanly inside painted road lines.
    Falls back to a single-lane estimate when only one ego boundary is detected.
    """
    ego_left, ego_right = find_ego_lanes(proposals)

    if ego_left is None and ego_right is None:
        return None

    anchor_len = 20
    model_w, model_h = img_size
    target_w, target_h = target_size
    scale_x = target_w / float(model_w)
    scale_y = target_h / float(model_h)

    if ego_left is not None and ego_right is not None:
        xs_l, ys_l, zs_l, vis_l = parse_lane_components(ego_left, anchor_len)
        xs_r, ys_r, zs_r, vis_r = parse_lane_components(ego_right, anchor_len)
        common_vis = vis_l & vis_r

        if common_vis.sum() < 2:
            common_vis = vis_l | vis_r
            if common_vis.sum() < 2:
                return None
            if vis_l.sum() >= vis_r.sum():
                use_l_xs = xs_l[common_vis] + left_margin
                use_l_zs = zs_l[common_vis]
                use_r_xs = xs_l[common_vis] + (STANDARD_LANE_WIDTH - left_margin - right_margin)
                use_r_zs = zs_l[common_vis]
            else:
                use_r_xs = xs_r[common_vis] - right_margin
                use_r_zs = zs_r[common_vis]
                use_l_xs = xs_r[common_vis] - (STANDARD_LANE_WIDTH - left_margin - right_margin)
                use_l_zs = zs_r[common_vis]
        else:
            use_l_xs = xs_l[common_vis] + left_margin
            use_l_zs = zs_l[common_vis]
            use_r_xs = xs_r[common_vis] - right_margin
            use_r_zs = zs_r[common_vis]

        ys = ys_l[common_vis] if ys_l is not None else ANCHOR_Y_STEPS[common_vis]

    elif ego_left is not None:
        xs_l, ys_l, zs_l, vis_l = parse_lane_components(ego_left, anchor_len)
        common_vis = vis_l
        if common_vis.sum() < 2:
            return None
        ys = ys_l[common_vis]
        use_l_xs = xs_l[common_vis] + left_margin
        use_l_zs = zs_l[common_vis]
        use_r_xs = xs_l[common_vis] + (STANDARD_LANE_WIDTH - left_margin - right_margin)
        use_r_zs = zs_l[common_vis]

    else:
        xs_r, ys_r, zs_r, vis_r = parse_lane_components(ego_right, anchor_len)
        common_vis = vis_r
        if common_vis.sum() < 2:
            return None
        ys = ys_r[common_vis]
        use_r_xs = xs_r[common_vis] - right_margin
        use_r_zs = zs_r[common_vis]
        use_l_xs = xs_r[common_vis] - (STANDARD_LANE_WIDTH - left_margin - right_margin)
        use_l_zs = zs_r[common_vis]

    def _project(xs, ys, zs):
        ones  = np.ones((1, len(zs)))
        coords = np.vstack((xs, ys, zs, ones))
        trans  = P_matrix @ coords
        u = trans[0, :] / (trans[2, :] + 1e-8)
        v = trans[1, :] / (trans[2, :] + 1e-8)
        return [(int(ui * scale_x), int(vi * scale_y)) for ui, vi in zip(u, v) if 0 <= ui < model_w and 0 <= vi < model_h]

    left_pts  = _project(use_l_xs, ys, use_l_zs)
    right_pts = _project(use_r_xs, ys, use_r_zs)

    if len(left_pts) < 2 or len(right_pts) < 2:
        return None

    poly_pts = left_pts + right_pts[::-1]
    return np.array(poly_pts, dtype=np.int32)

