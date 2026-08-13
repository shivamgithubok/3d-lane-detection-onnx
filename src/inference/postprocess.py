import numpy as np

from src.inference import lane_filter_config as cfg

ANCHOR_LEN = 20
ANCHOR_Y_STEPS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    dtype=np.float64,
)


def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _lane_geometry_ok(
    proposal,
    min_visible_points=None,
    max_abs_mean_x=None,
    max_lateral_jump=None,
    max_abs_slope=None,
):
    """Reject short, far, zig-zag, or steeply diagonal lane proposals."""
    min_visible_points = cfg.MIN_VISIBLE_POINTS if min_visible_points is None else min_visible_points
    max_abs_mean_x = cfg.MAX_ABS_MEAN_X_M if max_abs_mean_x is None else max_abs_mean_x
    max_lateral_jump = cfg.MAX_LATERAL_JUMP_M if max_lateral_jump is None else max_lateral_jump
    max_abs_slope = cfg.MAX_ABS_SLOPE if max_abs_slope is None else max_abs_slope

    xs = proposal[5 : 5 + ANCHOR_LEN].astype(np.float64)
    vis = proposal[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
    if int(vis.sum()) < min_visible_points:
        return False

    xs_v = xs[vis]
    ys_v = ANCHOR_Y_STEPS[vis]
    if abs(float(np.mean(xs_v))) > max_abs_mean_x:
        return False

    if len(xs_v) >= 2:
        jumps = np.abs(np.diff(xs_v))
        if float(np.max(jumps)) > max_lateral_jump:
            return False
        dy = float(ys_v[-1] - ys_v[0])
        if abs(dy) > 1e-3:
            slope = abs(float(xs_v[-1] - xs_v[0]) / dy)
            if slope > max_abs_slope:
                return False
    return True


def postprocess_onnx_output(
    reg_proposals,
    conf_threshold=None,
    nms_thres=None,
    max_lanes=None,
    apply_geometry_filter=True,
):
    """
    Decode Anchor3DLane raw proposals -> filtered lanes + scores.

    Defaults come from src.inference.lane_filter_config (easy to tune).
    """
    conf_threshold = cfg.CONF_THRESHOLD if conf_threshold is None else conf_threshold
    nms_thres = cfg.NMS_THRES_M if nms_thres is None else nms_thres
    max_lanes = cfg.MAX_LANES if max_lanes is None else max_lanes

    proposals = reg_proposals[0].copy()
    logits = softmax(proposals[:, 5 + 3 * ANCHOR_LEN :], axis=1)
    score = 1 - logits[:, 0]
    proposals[:, 1] = score
    proposals[:, 5 + 3 * ANCHOR_LEN :] = logits

    keep = score > conf_threshold
    proposals, kept_scores = proposals[keep], score[keep]
    if proposals.shape[0] == 0:
        return proposals, None

    if apply_geometry_filter:
        geo_keep = np.array([_lane_geometry_ok(p) for p in proposals], dtype=bool)
        proposals, kept_scores = proposals[geo_keep], kept_scores[geo_keep]
        if proposals.shape[0] == 0:
            return proposals, None

    order = np.argsort(-kept_scores)
    proposals, kept_scores = proposals[order], kept_scores[order]

    # Greedy duplicate suppression by mean lateral distance (meters)
    lane_xs_all = proposals[:, 5 : 5 + ANCHOR_LEN]
    vis_all = proposals[:, 5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0

    suppressed = np.zeros(len(proposals), dtype=bool)
    final_idx = []
    for i in range(len(proposals)):
        if suppressed[i]:
            continue
        final_idx.append(i)
        for j in range(i + 1, len(proposals)):
            if suppressed[j]:
                continue
            common = vis_all[i] & vis_all[j]
            if common.sum() < 2:
                continue
            dist = np.mean(np.abs(lane_xs_all[i][common] - lane_xs_all[j][common]))
            if dist < nms_thres:
                suppressed[j] = True

    proposals = proposals[final_idx]
    kept_scores = kept_scores[final_idx]

    # Prefer near-ego lanes when capping: re-rank by |mean_x| among survivors,
    # but break ties with score so strong far lanes can still win a slot.
    if max_lanes and len(proposals) > max_lanes:
        mean_x = []
        for p in proposals:
            vis = p[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
            xs = p[5 : 5 + ANCHOR_LEN][vis]
            mean_x.append(float(np.mean(xs)) if vis.sum() else 1e9)
        mean_x = np.asarray(mean_x, dtype=np.float64)
        # lower |mean_x| first, then higher score
        rank = np.lexsort((-kept_scores, np.abs(mean_x)))
        pick = rank[:max_lanes]
        # keep score-descending order for callers
        pick = pick[np.argsort(-kept_scores[pick])]
        proposals = proposals[pick]
        kept_scores = kept_scores[pick]

    return proposals, kept_scores


def _projective_transformation(P, x, y, z):
    ones = np.ones((1, len(z)))
    coords = np.vstack((x, y, z, ones))
    trans = P @ coords
    u = trans[0, :] / (trans[2, :] + 1e-8)
    v = trans[1, :] / (trans[2, :] + 1e-8)
    return u, v


def decode_lane_pixels(proposal, P_matrix):
    if isinstance(proposal, np.ndarray) and proposal.ndim == 2 and proposal.shape[1] == 3:
        xs = proposal[:, 0].astype(np.float64)
        ys = proposal[:, 1].astype(np.float64)
        zs = proposal[:, 2].astype(np.float64)
    else:
        lane_xs = proposal[5 : 5 + ANCHOR_LEN]
        lane_zs = proposal[5 + ANCHOR_LEN : 5 + 2 * ANCHOR_LEN]
        lane_vis = proposal[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
        if lane_vis.sum() < 2:
            return []
        xs = lane_xs[lane_vis].astype(np.float64)
        ys = ANCHOR_Y_STEPS[lane_vis]
        zs = lane_zs[lane_vis].astype(np.float64)

    if len(xs) < 2:
        return []

    u, v = _projective_transformation(P_matrix, xs, ys, zs)
    return list(zip(u, v))
