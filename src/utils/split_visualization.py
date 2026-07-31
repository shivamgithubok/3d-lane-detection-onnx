import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.visualization import draw_bev
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels
from src.utils.drivable_area import extract_ego_corridor_3d, get_ego_corridor_2d_pixels, fill_missing_lane_gaps, find_ego_lanes

def draw_futuristic_corner_bbox(img, pt1, pt2, color, thickness=2, corner_len=14):
    """Renders futuristic cybernetic corner brackets around detected vehicle bounding boxes."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    c_len = min(corner_len, w // 4, h // 4)

    # Top-Left corner
    cv2.line(img, (x1, y1), (x1 + c_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color, thickness, cv2.LINE_AA)

    # Top-Right corner
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color, thickness, cv2.LINE_AA)

    # Bottom-Left corner
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 + c_len), color, thickness, cv2.LINE_AA)

    # Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color, thickness, cv2.LINE_AA)


def draw_front_view_cipo(frame, proposals, objects, cipo_obj, P_matrix, show_drivable=True):
    """
    Renders front camera view with Translucent Green Drivable Corridor (left margin 0.50m, right margin 1.00m),
    WITHOUT outer cyan line borders, hiding ego cyan lane lines, and showing cybernetic vehicle bounding boxes with pre-trained MiDaS depth.
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]

    scale_x = w_img / 480.0
    scale_y = h_img / 360.0

    # Interpolate missing intermediate lane lines if gap >= 5.0m
    if proposals is not None:
        proposals = fill_missing_lane_gaps(proposals)

    ego_left, ego_right = find_ego_lanes(proposals) if proposals is not None else (None, None)

    in_path_objs = [obj for obj in objects if obj['in_path']]
    min_dist_in_path = min([obj['Z_3d'] for obj in in_path_objs]) if in_path_objs else 999.0

    # 1. Render Translucent Drivable Area Corridor (Left margin = 0.50m, Right margin = 1.00m, NO outer border lines)
    if show_drivable and proposals is not None:
        poly_2d = get_ego_corridor_2d_pixels(proposals, P_matrix, img_size=(480, 360), target_size=(w_img, h_img), left_margin=0.50, right_margin=1.00)
        if poly_2d is not None and len(poly_2d) > 2:
            overlay = annotated.copy()
            # Emerald Green (0, 255, 128) for safe drivable path, Red (0, 30, 255) for danger <15m
            corridor_color = (0, 30, 255) if min_dist_in_path < 15.0 else (0, 255, 128)
            cv2.fillPoly(overlay, [poly_2d], corridor_color)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    # 2. Draw 3D Lane Lines for ADJACENT lanes ONLY (Hiding Ego Left and Ego Right lines as requested)
    if proposals is not None:
        for lane in proposals:
            # Skip drawing ego-left and ego-right lines
            if ego_left is not None and np.array_equal(lane, ego_left):
                continue
            if ego_right is not None and np.array_equal(lane, ego_right):
                continue

            pts = decode_lane_pixels(lane, P_matrix)
            draw_pts = [(int(u * scale_x), int(v * scale_y)) for u, v in pts if 0 <= u < 480 and 0 <= v < 360]
            for i in range(1, len(draw_pts)):
                cv2.line(annotated, draw_pts[i-1], draw_pts[i], (255, 180, 0), 2, cv2.LINE_AA)

    # 3. Draw Cyberpunk Bounding Boxes & Distance Telemetry for Vehicles
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        track_id = obj.get('track_id', -1)
        color = obj['color'] # Yellow for >15m in-path, Red for <15m danger, Cyan for adjacent
        dist_m = obj['Z_3d']

        # Semi-transparent box background tint
        box_overlay = annotated.copy()
        cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(box_overlay, 0.12, annotated, 0.88, 0, annotated)

        # Futuristic cybernetic corner brackets (thickness=2)
        draw_futuristic_corner_bbox(annotated, (x1, y1), (x2, y2), color, thickness=2, corner_len=14)

        # Clean label format: 'VEH 03 - 19.1m' or 'TRK 08 - 28.4m'
        class_code = "TRK" if "truck" in obj['label'].lower() or "bus" in obj['label'].lower() else "VEH"
        id_str = f" {track_id:02d}" if track_id > 0 else ""
        label_text = f"{class_code}{id_str} - {dist_m:.1f}m"

        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(annotated, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 8, y1), color, -1)
        cv2.putText(annotated, label_text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def draw_bev_cipo(proposals, objects, max_z=60.0, cipo_status="SAFE"):
    """
    Renders top-down Bird's Eye View (BEV) map showing 3D lane lines, drivable area, and object positions.
    """
    if proposals is not None:
        proposals = fill_missing_lane_gaps(proposals)

    in_path_objs = [obj for obj in objects if obj['in_path']]
    min_dist_in_path = min([obj['Z_3d'] for obj in in_path_objs]) if in_path_objs else 999.0
    status_bev = "DANGER" if min_dist_in_path < 15.0 else "SAFE"

    bev = draw_bev(proposals, ANCHOR_Y_STEPS, cipo_status=status_bev)
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
    resized_bev = cv2.resize(bev_view, (bev_w, main_h))

    top_split = np.hstack((resized_front, resized_bev))

    hud = np.ones((hud_height, target_w, 3), dtype=np.uint8) * 20

    cv2.putText(hud, f"PERFORMANCE: {fps_val:.1f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(hud, "ENGINE: Triple TensorRT FP16 + YOLO ByteTrack + MiDaS Depth", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    status_text = "MONITOR: DRIVABLE AREA SAFETY ACTIVE"
    color = (0, 255, 128)

    cv2.putText(hud, status_text, (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(hud, "RULES: <15m RED | 15-30m YELLOW | >30m GREEN", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    canvas = np.vstack((top_split, hud))
    return canvas
