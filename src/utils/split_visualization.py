import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.visualization import draw_bev
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels
from src.utils.drivable_area import extract_ego_corridor_3d, get_ego_corridor_2d_pixels, fill_missing_lane_gaps, find_ego_lanes
from src.utils.draw_3d_box import draw_3d_wireframe_box


# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL 2D DETECTION BOX RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pro_detection_box(img, overlay, x1, y1, x2, y2, track_id, label, dist_m, color, is_cipo=False):
    """
    Pro-designer 2D detection box rendered via single-pass overlay:
      • Semi-transparent fill tinted by risk color (drawn on overlay)
      • Outer glow ring for CIPO / danger targets (drawn on overlay)
      • Dark translucent label chip background (drawn on overlay)
      • Vector line work & crisp drop-shadowed text (drawn on img)
    """

    # ── 1. Semi-transparent fill (on overlay) ─────────────────────────────
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # ── 2. Outer glow ring for CIPO / danger targets (on overlay) ─────────
    if is_cipo:
        glow_col = (30, 30, 220)
        for expand in (5, 3, 1):
            cv2.rectangle(overlay, (x1 - expand, y1 - expand),
                                   (x2 + expand, y2 + expand), glow_col, 1)

    # ── 3. Thin 1px border ────────────────────────────────────────────────
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # ── 4. Corner accent brackets ─────────────────────────────────────────
    w_box = x2 - x1
    h_box = y2 - y1
    c_len   = max(10, min(22, w_box // 4, h_box // 4))
    c_thick = 2

    corners = [
        (x1, y1,  1,  1),   # Top-Left
        (x2, y1, -1,  1),   # Top-Right
        (x1, y2,  1, -1),   # Bottom-Left
        (x2, y2, -1, -1),   # Bottom-Right
    ]
    for (cx, cy, dx, dy) in corners:
        cv2.line(img, (cx, cy), (cx + dx * c_len, cy),           color, c_thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx,              cy + dy * c_len), color, c_thick, cv2.LINE_AA)

    # ── 5. Label chip ─────────────────────────────────────────────────────
    lbl_lower = label.lower()
    if "truck" in lbl_lower or "bus" in lbl_lower:
        cls_code = "TRUCK"
        cls_icon = "▲"
    elif "motorcycle" in lbl_lower or "bike" in lbl_lower:
        cls_code = "MOTO"
        cls_icon = "◈"
    else:
        cls_code = "CAR"
        cls_icon = "●"

    id_str = f"#{track_id:02d}" if (track_id is not None and track_id > 0) else "#--"
    row1   = f" {id_str}  {cls_icon} {cls_code} "
    row2   = f"  {dist_m:.1f} m  "

    font_r1 = cv2.FONT_HERSHEY_DUPLEX
    font_r2 = cv2.FONT_HERSHEY_SIMPLEX
    fs1, th1 = 0.38, 1
    fs2, th2 = 0.42, 1

    (tw1, th1_px), _ = cv2.getTextSize(row1, font_r1, fs1, th1)
    (tw2, th2_px), _ = cv2.getTextSize(row2, font_r2, fs2, th2)

    chip_w = max(tw1, tw2) + 12
    chip_h = th1_px + th2_px + 18

    chip_x = x1
    chip_y = y1 - chip_h - 3
    if chip_y < 2:
        chip_y = y2 + 3

    # Dark translucent chip background (on overlay)
    cv2.rectangle(overlay, (chip_x, chip_y),
                          (chip_x + chip_w, chip_y + chip_h), (10, 10, 16), -1)

    # Colored top accent stripe (3px)
    cv2.rectangle(img, (chip_x, chip_y),
                        (chip_x + chip_w, chip_y + 3), color, -1)

    # Row 1: ID + class name (soft white, with 1px drop-shadow)
    y_r1 = chip_y + 3 + th1_px + 3
    cv2.putText(img, row1, (chip_x + 6 + 1, y_r1 + 1), font_r1, fs1, (0, 0, 0),     th1, cv2.LINE_AA)
    cv2.putText(img, row1, (chip_x + 6,     y_r1),     font_r1, fs1, (220, 220, 220), th1, cv2.LINE_AA)

    # Row 2: distance (risk color, with 1px drop-shadow)
    y_r2 = y_r1 + th2_px + 5
    cv2.putText(img, row2, (chip_x + 6 + 1, y_r2 + 1), font_r2, fs2, (0, 0, 0), th2, cv2.LINE_AA)
    cv2.putText(img, row2, (chip_x + 6,     y_r2),     font_r2, fs2, color,       th2, cv2.LINE_AA)



# ─────────────────────────────────────────────────────────────────────────────

def draw_futuristic_corner_bbox(img, pt1, pt2, color, thickness=1, corner_len=14):
    """Renders futuristic cybernetic corner brackets around detected vehicle bounding boxes."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    c_len = min(corner_len, w // 4, h // 4)

    cv2.line(img, (x1, y1), (x1 + c_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 + c_len), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# LANE DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ANCHOR_LEN = 20

from src.utils.drivable_area import extract_ego_corridor_3d, get_ego_corridor_2d_pixels, fill_missing_lane_gaps, find_ego_lanes, parse_lane_components


def _get_lane_mean_x(lane, anchor_len=ANCHOR_LEN):
    if lane is None:
        return 0.0
    xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
    return float(np.mean(xs[vis])) if vis.sum() >= 2 else 0.0


def _draw_lane_line(img, pts, color, thickness):
    """BEV-style flat painted lane: continuous anti-aliased polyline, no node dots."""
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)



# ─────────────────────────────────────────────────────────────────────────────

def draw_front_view_cipo(
    frame,
    proposals,
    objects,
    cipo_obj,
    P_matrix,
    show_drivable=True,
    ego_left=None,
    ego_right=None,
    frame_transform=None,
    road_state_valid=True,
    left_corridor_3d=None,
    right_corridor_3d=None,
):
    """
    Renders front camera view with ultra-fast single-pass overlay blending:
      - Translucent Green Drivable Corridor (left/right symmetric inset from ego lanes)
      - 3D lane polylines projected with full P_matrix (model Z kept — calibrated look)
      - Professional 2D detection boxes with ID/class/distance label chips
    """
    annotated = frame.copy()
    overlay   = frame.copy()
    h_img, w_img = annotated.shape[:2]

    scale_x = w_img / 480.0
    scale_y = h_img / 360.0

    if proposals is not None:
        proposals = fill_missing_lane_gaps(proposals)

    if road_state_valid and ego_left is None and ego_right is None:
        ego_left, ego_right = find_ego_lanes(proposals) if proposals is not None else (None, None)

    in_path_objs     = [obj for obj in objects if obj['in_path']]
    min_dist_in_path = min([obj['Z_3d'] for obj in in_path_objs]) if in_path_objs else 999.0
    danger           = min_dist_in_path < 15.0

    # ── 1. Drivable area corridor fill (on overlay) ──────────────────────
    if show_drivable and road_state_valid and proposals is not None:
        poly_2d = get_ego_corridor_2d_pixels(
            proposals, P_matrix,
            img_size=(480, 360), target_size=(w_img, h_img),
            ego_left=ego_left, ego_right=ego_right,
            model_to_target=(frame_transform.model_to_source if frame_transform is not None else None),
            left_corridor_3d=left_corridor_3d,
            right_corridor_3d=right_corridor_3d,
        )
        if poly_2d is not None and len(poly_2d) > 2:
            corridor_color = (0, 30, 255) if danger else (0, 220, 100)
            cv2.fillPoly(overlay, [poly_2d], corridor_color)

    # ── 2. Detection box fills & chips (on overlay & annotated) ───────────
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        track_id = obj.get('track_id', -1)
        color    = obj['color']
        dist_m   = obj['Z_3d']
        is_cipo  = obj.get('is_cipo', False)

        _draw_pro_detection_box(
            annotated, overlay,
            x1, y1, x2, y2,
            track_id=track_id,
            label=obj['label'],
            dist_m=dist_m,
            color=color,
            is_cipo=is_cipo,
        )

    # ── 3. SINGLE PASS ALPHA BLEND FOR ALL OVERLAYS ───────────────────────
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    # ── 4. Calibrated 3D lane polylines (use model Z + P_matrix) ──────────
    if proposals is not None:
        sorted_lanes = sorted(proposals, key=lambda l: _get_lane_mean_x(l))

        ego_l_idx, ego_r_idx = None, None
        for idx, lane in enumerate(sorted_lanes):
            if ego_left  is not None and np.array_equal(lane, ego_left):  ego_l_idx = idx
            if ego_right is not None and np.array_equal(lane, ego_right): ego_r_idx = idx

        for idx, lane in enumerate(sorted_lanes):
            # Draw ego + adjacent; skip far clutter
            is_ego = (
                (ego_left is not None and np.array_equal(lane, ego_left))
                or (ego_right is not None and np.array_equal(lane, ego_right))
            )
            is_adj = (
                (ego_l_idx is not None and idx == ego_l_idx - 1)
                or (ego_r_idx is not None and idx == ego_r_idx + 1)
            )
            if not is_ego and not is_adj:
                # still draw farther lanes thinner
                mean_x = _get_lane_mean_x(lane)
                if abs(mean_x) > 6.0:
                    continue
                lane_color, thickness = (120, 160, 200), 1
            elif is_ego:
                lane_color, thickness = (255, 180, 0), 2  # BEV ego cyan/orange
            else:
                lane_color, thickness = (0, 215, 255), 2

            # flat_ground=False → keep calibrated height (do not zero Z)
            pts = decode_lane_pixels(lane, P_matrix, flat_ground=False)
            model_pts = np.asarray([(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360])
            if frame_transform is not None and len(model_pts) > 0:
                target_pts = frame_transform.model_to_source(model_pts)
                draw_pts = [
                    (int(round(u)), int(round(v)))
                    for u, v in target_pts
                    if 0 <= u < w_img and 0 <= v < h_img
                ]
            else:
                draw_pts = [(int(u * scale_x), int(v * scale_y)) for u, v in model_pts]
            if len(draw_pts) > 1:
                _draw_lane_line(annotated, draw_pts, lane_color, thickness)

    return annotated



def draw_bev_cipo(
    proposals,
    objects,
    max_z=60.0,
    cipo_status="SAFE",
    left_corridor_3d=None,
    right_corridor_3d=None,
):
    """
    Renders top-down Bird's Eye View (BEV) map showing 3D lane lines, drivable area, and object positions.
    """
    if proposals is not None:
        proposals = fill_missing_lane_gaps(proposals)

    in_path_objs = [obj for obj in objects if obj['in_path']]
    min_dist_in_path = min([obj['Z_3d'] for obj in in_path_objs]) if in_path_objs else 999.0
    status_bev = "DANGER" if min_dist_in_path < 15.0 else "SAFE"

    bev = draw_bev(
        proposals,
        ANCHOR_Y_STEPS,
        cipo_status=status_bev,
        left_corridor_3d=left_corridor_3d,
        right_corridor_3d=right_corridor_3d,
        allow_auto_corridor=False,
    )
    h_bev, w_bev = bev.shape[:2]

    def world_to_bev_px(x, y):
        px = int((x - (-8.0)) / (8.0 - (-8.0)) * w_bev)
        py = int(h_bev - (y - 0.0) / (80.0 - 0.0) * (h_bev - 60) - 40)
        return px, py

    for obj in objects:
        x_3d = obj['X_3d']
        y_3d = obj['Z_3d']
        if 0 < y_3d <= 80.0:
            px, py = world_to_bev_px(x_3d, y_3d)
            if 0 <= px < w_bev and 0 <= py < h_bev:
                color = obj['color']
                radius = 6 if obj['in_path'] else 4
                cv2.circle(bev, (px, py), radius, color, -1)

                track_id = obj.get('track_id', -1)
                id_str = f"#{track_id} " if track_id > 0 else ""
                label = f"{id_str}{obj['label'].upper()}"
                cv2.putText(bev, label, (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return bev


def create_split_window(front_view, bev_view, cipo_obj, fps_val, canvas_size=(720, 1080)):
    """
    Combines Front View, BEV Map, and Telemetry HUD into a unified multi-panel split window.
    """
    target_h, target_w = canvas_size
    hud_height = 80
    main_h = target_h - hud_height

    front_w = int(target_w * 0.6)
    bev_w = target_w - front_w

    resized_front = cv2.resize(front_view, (front_w, main_h))
    resized_bev   = cv2.resize(bev_view,   (bev_w,   main_h))

    top_split = np.hstack((resized_front, resized_bev))

    hud = np.ones((hud_height, target_w, 3), dtype=np.uint8) * 20
    cv2.putText(hud, f"PERFORMANCE: {fps_val:.1f} FPS", (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(hud, "ENGINE: Triple TensorRT FP16 + YOLO ByteTrack + MiDaS Depth", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.putText(hud, "MONITOR: DRIVABLE AREA SAFETY ACTIVE", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)
    cv2.putText(hud, "RULES: <15m RED | 15-30m YELLOW | >30m GREEN", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    canvas = np.vstack((top_split, hud))
    return canvas
