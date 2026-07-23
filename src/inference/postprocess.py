import numpy as np

ANCHOR_LEN = 20
ANCHOR_Y_STEPS = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], dtype=np.float64)

def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def postprocess_onnx_output(reg_proposals, conf_threshold=0.2, nms_thres=3.0):
    proposals = reg_proposals[0].copy()
    logits = softmax(proposals[:, 5 + 3 * ANCHOR_LEN:], axis=1)
    score = 1 - logits[:, 0]
    proposals[:, 1] = score
    proposals[:, 5 + 3 * ANCHOR_LEN:] = logits

    keep = score > conf_threshold
    proposals, kept_scores = proposals[keep], score[keep]
    if proposals.shape[0] == 0:
        return proposals, None

    order = np.argsort(-kept_scores)
    proposals, kept_scores = proposals[order], kept_scores[order]

    # real duplicate suppression: greedy, by mean lateral distance (meters),
    # matching the model's own configured nms_thres
    lane_xs_all = proposals[:, 5:5 + ANCHOR_LEN]
    vis_all = proposals[:, 5 + 2*ANCHOR_LEN:5 + 3*ANCHOR_LEN] > 0

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

    return proposals[final_idx], kept_scores[final_idx]

def _projective_transformation(P, x, y, z):
    ones = np.ones((1, len(z)))
    coords = np.vstack((x, y, z, ones))
    trans = P @ coords
    u = trans[0, :] / (trans[2, :] + 1e-8)
    v = trans[1, :] / (trans[2, :] + 1e-8)
    return u, v

def decode_lane_pixels(proposal, P_matrix):
    lane_xs = proposal[5:5 + ANCHOR_LEN]
    lane_zs = proposal[5 + ANCHOR_LEN:5 + 2 * ANCHOR_LEN]
    lane_vis = proposal[5 + 2 * ANCHOR_LEN:5 + 3 * ANCHOR_LEN] > 0
    if lane_vis.sum() < 2:
        return []
    xs = lane_xs[lane_vis].astype(np.float64)
    ys = ANCHOR_Y_STEPS[lane_vis]
    zs = lane_zs[lane_vis].astype(np.float64)
    u, v = _projective_transformation(P_matrix, xs, ys, zs)
    return list(zip(u, v))