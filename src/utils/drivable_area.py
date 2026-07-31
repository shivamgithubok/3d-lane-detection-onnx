import numpy as np
import cv2
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels

STANDARD_LANE_WIDTH = 3.7  # Standard highway lane width in meters

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
        lane_xs = lane[5:5 + anchor_len]
        lane_vis = lane[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
        if lane_vis.sum() < 2:
            continue
        valid_xs = lane_xs[lane_vis]
        mean_x = np.mean(valid_xs)

        if mean_x <= 0:
            left_lanes.append((abs(mean_x), lane))
        else:
            right_lanes.append((mean_x, lane))

    # Sort by proximity to vehicle center (X = 0)
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
        lane_xs = lane[5:5 + anchor_len]
        lane_vis = lane[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
        if lane_vis.sum() >= 2:
            mean_x = np.mean(lane_xs[lane_vis])
            valid_proposals.append((mean_x, lane))

    if len(valid_proposals) < 2:
        return proposals

    # Sort proposals laterally from left to right by mean X coordinate
    valid_proposals = sorted(valid_proposals, key=lambda item: item[0])
    augmented_proposals = [item[1] for item in valid_proposals]

    # Inspect gaps between adjacent detected lanes
    i = 0
    while i < len(augmented_proposals) - 1:
        lane_curr = augmented_proposals[i]
        lane_next = augmented_proposals[i + 1]

        xs_curr = lane_curr[5:5 + anchor_len]
        xs_next = lane_next[5:5 + anchor_len]

        vis_curr = lane_curr[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
        vis_next = lane_next[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0

        common_vis = vis_curr & vis_next
        if common_vis.sum() >= 2:
            gap_m = np.mean(xs_next[common_vis] - xs_curr[common_vis])
            if gap_m >= min_gap:
                num_missing = int(round(gap_m / standard_lane_width)) - 1
                for step in range(1, num_missing + 1):
                    shift_offset = step * standard_lane_width
                    synth_lane = lane_curr.copy()
                    synth_lane[5:5 + anchor_len] = xs_curr + shift_offset
                    augmented_proposals.insert(i + step, synth_lane)
                i += num_missing
        i += 1

    return np.array(augmented_proposals)


def extract_ego_corridor_3d(proposals, anchor_len=20, left_margin=0.50, right_margin=1.00):
    """
    Extracts matching (X, Y, Z) 3D coordinate arrays for Ego-Left and Ego-Right lanes,
    applying left_margin=0.50m and right_margin=1.00m.
    """
    ego_left, ego_right = find_ego_lanes(proposals, anchor_len)

    if ego_left is None and ego_right is None:
        return None, None

    y_steps = ANCHOR_Y_STEPS

    if ego_left is not None:
        xs_l = ego_left[5:5 + anchor_len] + left_margin
        zs_l = ego_left[5 + anchor_len:5 + 2 * anchor_len]
        vis_l = ego_left[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
    else:
        xs_l, zs_l, vis_l = None, None, None

    if ego_right is not None:
        xs_r = ego_right[5:5 + anchor_len] - right_margin
        zs_r = ego_right[5 + anchor_len:5 + 2 * anchor_len]
        vis_r = ego_right[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
    else:
        xs_r, zs_r, vis_r = None, None, None

    left_pts = []
    right_pts = []

    for i in range(anchor_len):
        y_m = y_steps[i]

        has_l = vis_l[i] if vis_l is not None else False
        has_r = vis_r[i] if vis_r is not None else False

        if has_l and has_r:
            left_pts.append((xs_l[i], y_m, zs_l[i]))
            right_pts.append((xs_r[i], y_m, zs_r[i]))
        elif has_l:
            xl, zl = xs_l[i], zs_l[i]
            left_pts.append((xl, y_m, zl))
            left_pts.append((xl + STANDARD_LANE_WIDTH - (left_margin + right_margin), y_m, zl))
        elif has_r:
            xr, zr = xs_r[i], zs_r[i]
            left_pts.append((xr - STANDARD_LANE_WIDTH + (left_margin + right_margin), y_m, zr))
            right_pts.append((xr, y_m, zr))

    if len(left_pts) < 2 or len(right_pts) < 2:
        return None, None

    return np.array(left_pts), np.array(right_pts)


def get_ego_corridor_2d_pixels(proposals, P_matrix, img_size=(480, 360), target_size=(1080, 720), left_margin=0.50, right_margin=1.00):
    """
    Projects Ego-Left and Ego-Right 3D lane lines directly onto 2D camera pixels,
    applying left_margin=0.50m and right_margin=1.00m to sit cleanly inside painted road lines.
    """
    ego_left, ego_right = find_ego_lanes(proposals)
    if ego_left is None or ego_right is None:
        return None

    model_w, model_h = img_size
    target_w, target_h = target_size
    scale_x = target_w / float(model_w)
    scale_y = target_h / float(model_h)

    vis_l = ego_left[5 + 40:5 + 60] > 0
    vis_r = ego_right[5 + 40:5 + 60] > 0
    common_vis = vis_l & vis_r

    if common_vis.sum() < 2:
        return None

    xs_l = ego_left[5:5 + 20][common_vis] + left_margin
    zs_l = ego_left[5 + 20:5 + 40][common_vis]

    xs_r = ego_right[5:5 + 20][common_vis] - right_margin
    zs_r = ego_right[5 + 20:5 + 40][common_vis]

    ys = ANCHOR_Y_STEPS[common_vis]

    def _project(xs, ys, zs):
        ones = np.ones((1, len(zs)))
        coords = np.vstack((xs, ys, zs, ones))
        trans = P_matrix @ coords
        u = trans[0, :] / (trans[2, :] + 1e-8)
        v = trans[1, :] / (trans[2, :] + 1e-8)
        return [(int(ui * scale_x), int(vi * scale_y)) for ui, vi in zip(u, v) if 0 <= ui < model_w and 0 <= vi < model_h]

    left_pts = _project(xs_l, ys, zs_l)
    right_pts = _project(xs_r, ys, zs_r)

    if len(left_pts) < 2 or len(right_pts) < 2:
        return None

    # Closed polygon loop: Forward along left lane points, Backward along right lane points
    poly_pts = left_pts + right_pts[::-1]
    return np.array(poly_pts, dtype=np.int32)
