import numpy as np
from src.utils.drivable_area import find_ego_lanes
from src.inference.postprocess import decode_lane_pixels

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

    def get_2d_lane_u_at_v(self, lane_proposal, v_target):
        """
        Calculates the 2D projected u-pixel coordinate of a lane line at a specific v-pixel row.
        """
        pts_2d = decode_lane_pixels(lane_proposal, self.P)
        if len(pts_2d) < 2:
            return None

        us = [p[0] for p in pts_2d]
        vs = [p[1] for p in pts_2d]

        # Sort by v ascending
        order = np.argsort(vs)
        vs = np.array(vs)[order]
        us = np.array(us)[order]

        if v_target < vs[0] or v_target > vs[-1]:
            return None

        u_interp = float(np.interp(v_target, vs, us))
        return u_interp

    def process_detections(self, detections, lane_proposals, frame_size=(1080, 720), depth_map=None, depth_estimator=None):
        processed_objects = []
        w_img, h_img = frame_size
        scale_u = 480.0 / float(w_img)
        scale_v = 360.0 / float(h_img)

        ego_left, ego_right = find_ego_lanes(lane_proposals, ANCHOR_LEN)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            u_img = (x1 + x2) / 2.0
            v_img = float(y2)

            # Scale 2D pixel coordinates to model space (480x360) BEFORE projecting to 3D ground!
            u_model = u_img * scale_u
            v_model = v_img * scale_v

            # Query TensorRT Monocular Depth Engine for exact metric distance Z
            if depth_map is not None and depth_estimator is not None:
                Y_3d = depth_estimator.query_vehicle_depth(depth_map, det['bbox'], w_img, h_img)
                denom_x = self.P[0, 0]
                X_3d = float((Y_3d * (self.P[2, 1] * u_model - self.P[0, 1])) / denom_x)
            else:
                X_3d, Y_3d = self.project_2d_to_3d_ground(u_model, v_model)

            if Y_3d <= 0 or Y_3d > 100.0:
                continue

            # 2D Camera-Space Lane Association Rule
            u_left_2d = self.get_2d_lane_u_at_v(ego_left, v_model) if ego_left is not None else None
            u_right_2d = self.get_2d_lane_u_at_v(ego_right, v_model) if ego_right is not None else None

            # Determine lane membership directly in 2D Camera View
            is_left_of_left_lane = (u_left_2d is not None) and (u_model < u_left_2d - 5.0)
            is_right_of_right_lane = (u_right_2d is not None) and (u_model > u_right_2d + 5.0)

            if is_left_of_left_lane:
                in_path = False
                X_3d = min(X_3d, -2.40)
            elif is_right_of_right_lane:
                in_path = False
                X_3d = max(X_3d, +2.40)
            else:
                # Check strictly inside 3D Ego Drivable Corridor
                in_path = abs(X_3d) <= 1.50

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
                'Z_3d': Y_3d, # Metric forward distance in meters
                'in_path': in_path,
                'status': status,
                'color': color,
                'is_cipo': in_path and Y_3d < self.danger_dist
            }

            processed_objects.append(obj_info)

        return processed_objects, None
