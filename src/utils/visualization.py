import numpy as np
import cv2

BEV_CANVAS_SIZE = (600, 400)   # (height, width) in pixels
METERS_PER_PIXEL = 0.25         # zoom level: how many meters each pixel represents
X_RANGE = (-15, 15)              # lateral, meters
Y_RANGE = (0, 100)                # forward, meters (matches anchor_y_steps)

def world_to_canvas(x, y):
    """Real-world (x=lateral meters, y=forward meters) -> BEV canvas pixel."""
    px = int((x - X_RANGE[0]) / METERS_PER_PIXEL)
    py = int(BEV_CANVAS_SIZE[0] - (y - Y_RANGE[0]) / METERS_PER_PIXEL)
    return px, py

def draw_bev(proposals, anchor_y_steps, anchor_len=20):
    canvas = np.zeros((BEV_CANVAS_SIZE[0], BEV_CANVAS_SIZE[1], 3), dtype=np.uint8)

    # draw reference gridlines every 5 meters, forward and lateral
    for y_m in range(0, 101, 20):
        py = int(BEV_CANVAS_SIZE[0] - (y_m - Y_RANGE[0]) / METERS_PER_PIXEL)
        cv2.line(canvas, (0, py), (BEV_CANVAS_SIZE[1], py), (40, 40, 40), 1)
        cv2.putText(canvas, f"{y_m}m", (5, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    ego_x, _ = world_to_canvas(0, 0)
    cv2.line(canvas, (ego_x, 0), (ego_x, BEV_CANVAS_SIZE[0]), (40, 40, 40), 1)

    if proposals is not None:
        for lane in proposals:
            lane_xs = lane[5:5 + anchor_len]
            lane_vis = lane[5 + 2*anchor_len:5 + 3*anchor_len] > 0
            pts = [world_to_canvas(lane_xs[i], anchor_y_steps[i]) for i in range(anchor_len) if lane_vis[i]]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(canvas, p, 3, (0, 200, 255), -1)

    # mark ego vehicle position
    cv2.circle(canvas, (ego_x, BEV_CANVAS_SIZE[0] - 5), 5, (255, 0, 0), -1)
    return canvas