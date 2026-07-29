import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.visualization import draw_bev
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels

def draw_front_view_cipo(frame, proposals, objects, cipo_obj, P_matrix):
    """
    Renders front camera view with properly scaled 3D projected lanes, tracked cars/trucks, and CIPO alerts.
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]

    # Coordinate scaling from model space (480x360) to camera image space (w_img, h_img)
    scale_x = w_img / 480.0
    scale_y = h_img / 360.0

    # 1. Draw 3D Lane Lines with correct resolution scaling
    if proposals is not None:
        for lane in proposals:
            pts = decode_lane_pixels(lane, P_matrix)
            draw_pts = [(int(u * scale_x), int(v * scale_y)) for u, v in pts if 0 <= u < 480 and 0 <= v < 360]
            for i in range(1, len(draw_pts)):
                cv2.line(annotated, draw_pts[i-1], draw_pts[i], (0, 255, 0), 3, cv2.LINE_AA)

    # 2. Draw Bounding Boxes for Tracked Cars & Trucks (ByteTrack)
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        color = obj['color']
        is_cipo = obj['is_cipo']
        thickness = 3 if is_cipo else 2
        track_id = obj.get('track_id', -1)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        id_str = f"#{track_id} " if track_id > 0 else ""
        label_text = f"{id_str}{obj['label'].upper()} {obj['Z_3d']:.1f}m"
        if is_cipo:
            label_text = f"CIPO {id_str}{obj['label'].upper()}: {obj['Z_3d']:.1f}m [{obj['status']}]"

        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
        cv2.putText(annotated, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        u_bot = int((x1 + x2) / 2)
        cv2.circle(annotated, (u_bot, y2), 5, color, -1)

    # 3. Alert Header Overlay
    if cipo_obj is not None:
        status = cipo_obj['status']
        dist = cipo_obj['Z_3d']
        track_id = cipo_obj.get('track_id', -1)
        id_str = f"#{track_id} " if track_id > 0 else ""

        if status == "DANGER":
            alert_text = f"CRITICAL FCW ALERT: {cipo_obj['label'].upper()} {id_str}{dist:.1f}m AHEAD IN LANE!"
            bg_color = (0, 0, 220)
        elif status == "WARNING":
            alert_text = f"CIPO WARNING: {cipo_obj['label'].upper()} {id_str}IN LANE AT {dist:.1f}m"
            bg_color = (0, 180, 220)
        else:
            alert_text = f"CIPO TRACKED: {cipo_obj['label'].upper()} {id_str}{dist:.1f}m (SAFE)"
            bg_color = (0, 150, 0)

        cv2.rectangle(annotated, (0, 0), (w_img, 36), bg_color, -1)
        cv2.putText(annotated, alert_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(annotated, (0, 0), (w_img, 36), (40, 40, 40), -1)
        cv2.putText(annotated, "CIPO STATUS: LANE CLEAR (NO TARGET IN ROI)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def draw_bev_cipo(proposals, objects, max_z=60.0):
    """
    Renders top-down Bird's Eye View (BEV) map showing 3D lane lines and object positions.
    """
    bev = draw_bev(proposals, ANCHOR_Y_STEPS)
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
                radius = 8 if obj['is_cipo'] else 5
                thickness = -1 if obj['is_cipo'] else 2
                cv2.circle(bev, (px, py), radius, color, thickness)

                track_id = obj.get('track_id', -1)
                id_str = f"#{track_id} " if track_id > 0 else ""
                label = f"{id_str}{y_3d:.1f}m"
                if obj['is_cipo']:
                    label = f"CIPO {id_str}{y_3d:.1f}m"
                    ego_px, ego_py = world_to_bev_px(0, 0)
                    cv2.line(bev, (ego_px, ego_py), (px, py), color, 1, cv2.LINE_AA)

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

    hud = np.ones((hud_height, target_w, 3), dtype=np.uint8) * 30

    cv2.putText(hud, f"PERFORMANCE: {fps_val:.1f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(hud, "ENGINE: TensorRT FP16 + YOLO ByteTrack", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if cipo_obj is not None:
        track_id = cipo_obj.get('track_id', -1)
        id_str = f"#{track_id} " if track_id > 0 else ""
        status_str = f"CIPO TARGET: {cipo_obj['label'].upper()} {id_str}[{cipo_obj['status']}]"
        dist_str = f"RANGE: {cipo_obj['Z_3d']:.1f} meters"
        pos_str = f"X-OFFSET: {cipo_obj['X_3d']:+.2f} m"
        color = cipo_obj['color']
    else:
        status_str = "STATUS: NO IN-PATH TARGET"
        dist_str = "RANGE: --"
        pos_str = "X-OFFSET: --"
        color = (180, 180, 180)

    cv2.putText(hud, status_str, (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(hud, dist_str, (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(hud, pos_str, (580, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    canvas = np.vstack((top_split, hud))
    return canvas
