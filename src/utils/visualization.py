import numpy as np
import cv2
from src.utils.drivable_area import extract_ego_corridor_3d

# BEV Canvas dimensions (Height x Width)
BEV_HEIGHT = 700
BEV_WIDTH = 500
BEV_CANVAS_SIZE = (BEV_HEIGHT, BEV_WIDTH)

# Realistic lateral & forward coordinate bounds (in meters)
X_RANGE = (-8.0, 8.0)
Y_RANGE = (0.0, 80.0)

def world_to_canvas(x, y):
    """Real-world (x=lateral meters, y=forward meters) -> BEV canvas pixel coordinate."""
    px = int((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * BEV_WIDTH)
    py = int(BEV_HEIGHT - (y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * (BEV_HEIGHT - 60) - 40)
    return px, py

def draw_bev(
    proposals,
    anchor_y_steps,
    anchor_len=20,
    cipo_status="SAFE",
    left_corridor_3d=None,
    right_corridor_3d=None,
    allow_auto_corridor=True,
):
    """
    Renders a clean, realistic top-down Bird's Eye View (BEV) map with drivable corridor and 2px lane lines.
    """
    # 1. Dark sleek road background
    canvas = np.full((BEV_HEIGHT, BEV_WIDTH, 3), (22, 27, 34), dtype=np.uint8)

    # 2. Driveable road asphalt corridor (-6m to +6m)
    left_road_x, _ = world_to_canvas(-6.0, 0)
    right_road_x, _ = world_to_canvas(6.0, 0)
    cv2.rectangle(canvas, (left_road_x, 0), (right_road_x, BEV_HEIGHT), (30, 35, 43), -1)

    # 3. Draw Drivable Area Corridor Polygon (Green for Safe / Red for Danger)
    if left_corridor_3d is None and right_corridor_3d is None and allow_auto_corridor:
        left_3d, right_3d = extract_ego_corridor_3d(proposals, anchor_len)
    else:
        left_3d, right_3d = left_corridor_3d, right_corridor_3d
    if left_3d is not None and right_3d is not None:
        pts_left_bev = [world_to_canvas(x, y) for x, y, _ in left_3d]
        pts_right_bev = [world_to_canvas(x, y) for x, y, _ in right_3d]
        poly_bev = np.array(pts_left_bev + pts_right_bev[::-1], dtype=np.int32)

        overlay = canvas.copy()
        # Dynamic corridor color: Red for DANGER (<15m), Green (0, 255, 128) for SAFE
        corridor_color = (0, 30, 220) if cipo_status == "DANGER" else (0, 255, 128)
        cv2.fillPoly(overlay, [poly_bev], corridor_color)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # 4. Longitudinal distance gridlines (every 10 meters)
    for y_m in range(10, 81, 10):
        _, py = world_to_canvas(0, y_m)
        cv2.line(canvas, (left_road_x, py), (right_road_x, py), (48, 54, 65), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{y_m}m", (12, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 145), 1, cv2.LINE_AA)

    # 5. Lateral position gridlines (every 2 meters)
    for x_m in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        px, _ = world_to_canvas(x_m, 0)
        line_color = (65, 75, 90) if x_m != 0.0 else (90, 105, 125)
        thickness = 1 if x_m != 0.0 else 2
        cv2.line(canvas, (px, 20), (px, BEV_HEIGHT - 30), line_color, thickness, cv2.LINE_AA)
        if x_m != 0.0:
            cv2.putText(canvas, f"{x_m:+.1f}m", (px - 15, BEV_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 130, 145), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "0m (Ego)", (px - 22, BEV_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)

    # 6. Draw sleek 2px 3D detected lane lines (thickness=2)
    if proposals is not None and len(proposals) > 0:
        for idx, lane in enumerate(proposals):
            if isinstance(lane, np.ndarray) and lane.ndim == 2 and lane.shape[1] == 3:
                pts_world = [(lane[i, 0], lane[i, 1]) for i in range(len(lane))]
            else:
                lane_xs = lane[5:5 + anchor_len]
                lane_vis = lane[5 + 2*anchor_len:5 + 3*anchor_len] > 0
                pts_world = [(lane_xs[i], anchor_y_steps[i]) for i in range(anchor_len) if lane_vis[i]]

            # Map to canvas pixels
            pts_canvas = [world_to_canvas(wx, wy) for wx, wy in pts_world]
            valid_pts = [(px, py) for px, py in pts_canvas if 0 <= px < BEV_WIDTH and 0 <= py < BEV_HEIGHT]

            if len(valid_pts) > 1:
                mean_x = np.mean([wx for wx, wy in pts_world])
                if abs(mean_x) < 2.0:
                    lane_color = (255, 180, 0) # Ego lane lines (Electric Cyan)
                elif mean_x < 0:
                    lane_color = (0, 215, 255) # Left adjacent lane (Gold)
                else:
                    lane_color = (255, 200, 0) # Right adjacent lane (Light Cyan)

                # Draw sleek 2px lane polyline (thickness=2)
                for i in range(1, len(valid_pts)):
                    cv2.line(canvas, valid_pts[i-1], valid_pts[i], lane_color, 2, cv2.LINE_AA)

                # Draw node points
                for p in valid_pts[::2]:
                    cv2.circle(canvas, p, 2, (255, 255, 255), -1, cv2.LINE_AA)

    # 7. Render Ego Vehicle Icon at bottom center
    ego_px, ego_py = world_to_canvas(0.0, 0.0)
    car_w, car_h = 24, 40
    top_left = (ego_px - car_w // 2, ego_py - car_h // 2)
    bottom_right = (ego_px + car_w // 2, ego_py + car_h // 2)
    cv2.rectangle(canvas, top_left, bottom_right, (235, 140, 0), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2, cv2.LINE_AA)
    # Vehicle direction arrow
    cv2.arrowedLine(canvas, (ego_px, ego_py + 10), (ego_px, ego_py - 15), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)

    # Title header
    cv2.putText(canvas, "BIRD'S EYE VIEW (BEV)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas
