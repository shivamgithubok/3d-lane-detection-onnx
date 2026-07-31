import numpy as np
from src.utils.drivable_area import find_ego_lanes

# Camera Projection Matrix for cam_height = 1.5m and pitch = -3 degrees
DEFAULT_P_MATRIX = np.array([
    [503.75, 239.67108834, 12.5606295, 0.0],
    [0.0, 181.326628, -557.993558, 850.078125],
    [0.0, 0.998629535, 0.0523359562, 0.0]
])

ANCHOR_LEN = 20
ANCHOR_Y_STEPS = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], dtype=np.float64)

class CIPOTracker:
    def __init__(self, P_matrix=DEFAULT_P_MATRIX, danger_dist=15.0, warning_dist=30.0):
        self.P = P_matrix
        self.danger_dist = danger_dist  # < 15m DANGER (Red)
        self.warning_dist = warning_dist # 15m - 30m WARNING (Yellow)

    def project_2d_to_3d_ground(self, u, v):
        """
        Projects 2D pixel (u, v) in model space (480x360) to 3D ground coordinates (X_meters, Y_meters).
        """
        P = self.P
        denom_y = (P[2, 1] * v - P[1, 1])
        if abs(denom_y) < 1e-4:
            Y = 50.0
        else:
            Y = float(P[1, 3] / denom_y)

        Y = max(1.0, min(100.0, Y)) # Clamp to valid range

        # Lateral X coordinate
        denom_x = P[0, 0]
        X = float((Y * (P[2, 1] * u - P[0, 1])) / denom_x)
        return X, Y

    def is_inside_drivable_area(self, X_obj, Y_obj, lane_proposals):
        """
        Checks if 3D coordinate (X_obj, Y_obj) lies strictly inside the 3D Ego Drivable Area Corridor.
        Uses inner lateral threshold to prevent adjacent lane vehicles from triggering false in-path status.
        """
        ego_left, ego_right = find_ego_lanes(lane_proposals, ANCHOR_LEN)
        
        if ego_left is None and ego_right is None:
            return abs(X_obj) <= 1.25 and 0 < Y_obj <= 80.0

        X_left, X_right = None, None

        if ego_left is not None:
            xs_l = ego_left[5:5 + ANCHOR_LEN]
            vis_l = ego_left[5 + 2 * ANCHOR_LEN:5 + 3 * ANCHOR_LEN] > 0
            if vis_l.sum() >= 2:
                X_left = float(np.interp(Y_obj, ANCHOR_Y_STEPS[vis_l], xs_l[vis_l]))

        if ego_right is not None:
            xs_r = ego_right[5:5 + ANCHOR_LEN]
            vis_r = ego_right[5 + 2 * ANCHOR_LEN:5 + 3 * ANCHOR_LEN] > 0
            if vis_r.sum() >= 2:
                X_right = float(np.interp(Y_obj, ANCHOR_Y_STEPS[vis_r], xs_r[vis_r]))

        # Inner margin to avoid false positive triggers from adjacent lane vehicles
        inner_margin = -0.30
        if X_left is not None and X_right is not None:
            X_min = min(X_left, X_right) - inner_margin
            X_max = max(X_left, X_right) + inner_margin
        elif X_left is not None:
            X_min = X_left - inner_margin
            X_max = X_left + 3.2
        elif X_right is not None:
            X_min = X_right - 3.2
            X_max = X_right + inner_margin
        else:
            X_min, X_max = -1.25, 1.25

        return X_min <= X_obj <= X_max

    def process_detections(self, detections, lane_proposals, frame_size=(1080, 720)):
        processed_objects = []
        w_img, h_img = frame_size
        scale_u = 480.0 / float(w_img)
        scale_v = 360.0 / float(h_img)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            u_img = (x1 + x2) / 2.0
            v_img = float(y2)

            # Scale 2D pixel coordinates to model space (480x360) BEFORE projecting to 3D ground!
            u_model = u_img * scale_u
            v_model = v_img * scale_v

            X_3d, Y_3d = self.project_2d_to_3d_ground(u_model, v_model)

            if Y_3d <= 0 or Y_3d > 100.0:
                continue

            # Check strictly inside Drivable Area Corridor
            in_path = self.is_inside_drivable_area(X_3d, Y_3d, lane_proposals)

            if not in_path:
                status = "OUT OF PATH"
                color = (255, 220, 0) # Electric Neon Cyan for adjacent / out of path vehicles
            elif Y_3d < self.danger_dist: # < 15m DANGER (RED)
                status = "DANGER <15m"
                color = (0, 0, 255) # RED for critical danger <15m inside drivable corridor
            else: # > 15m IN PATH (YELLOW)
                status = f"IN PATH ({Y_3d:.1f}m)"
                color = (0, 215, 255) # YELLOW for vehicles inside drivable area at >15m distance!

            obj_info = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': det.get('class', 'car'),
                'track_id': det.get('track_id', -1),
                'conf': det.get('conf', 1.0),
                'X_3d': X_3d,
                'Z_3d': Y_3d, # Forward distance in meters
                'in_path': in_path,
                'status': status,
                'color': color,
                'is_cipo': in_path and Y_3d < self.danger_dist
            }

            processed_objects.append(obj_info)

        return processed_objects, None
