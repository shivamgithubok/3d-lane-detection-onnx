import numpy as np

# Camera Calibration Matrix for cam_height = 1.5m and pitch = -3 degrees
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
        self.danger_dist = danger_dist
        self.warning_dist = warning_dist

    def project_2d_to_3d_ground(self, u, v):
        """
        Projects 2D pixel (u, v) on front camera image to 3D ground coordinates (X_meters, Y_meters).
        Uses camera calibration matrix P matching Anchor3DLane model.
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

    def is_inside_lane_roi(self, X_obj, Y_obj, lane_proposals):
        """
        Checks if 3D coordinate (X_obj, Y_obj) lies between left and right lane boundaries.
        """
        if lane_proposals is None or len(lane_proposals) == 0:
            return abs(X_obj) <= 2.0 and 0 < Y_obj <= 80.0

        # Extract 3D X coordinates across anchor steps for each detected lane
        valid_lanes = []
        for lane in lane_proposals:
            if len(lane) >= 5 + 3 * ANCHOR_LEN:
                lane_xs = lane[5:5 + ANCHOR_LEN]
                lane_vis = lane[5 + 2 * ANCHOR_LEN:5 + 3 * ANCHOR_LEN] > 0
                if lane_vis.sum() >= 2:
                    valid_lanes.append((lane_xs, lane_vis))

        if len(valid_lanes) == 0:
            return abs(X_obj) <= 2.0 and 0 < Y_obj <= 80.0

        # Sort lanes left to right by mean X coordinate
        valid_lanes.sort(key=lambda item: np.mean(item[0][item[1]]))

        left_lane = None
        right_lane = None

        for i in range(len(valid_lanes) - 1):
            l1_mean = np.mean(valid_lanes[i][0][valid_lanes[i][1]])
            l2_mean = np.mean(valid_lanes[i+1][0][valid_lanes[i+1][1]])
            if l1_mean <= 0.5 and l2_mean >= -0.5:
                left_lane = valid_lanes[i]
                right_lane = valid_lanes[i+1]
                break

        if left_lane is None or right_lane is None:
            left_lane = valid_lanes[0]
            right_lane = valid_lanes[-1]

        # Interpolate X_left and X_right at forward distance Y_obj
        left_xs, left_vis = left_lane
        right_xs, right_vis = right_lane

        X_left_interp = np.interp(Y_obj, ANCHOR_Y_STEPS[left_vis], left_xs[left_vis])
        X_right_interp = np.interp(Y_obj, ANCHOR_Y_STEPS[right_vis], right_xs[right_vis])

        margin = 0.35
        is_in = (min(X_left_interp, X_right_interp) - margin) <= X_obj <= (max(X_left_interp, X_right_interp) + margin)
        return is_in

    def process_detections(self, detections, lane_proposals):
        processed_objects = []
        cipo_obj = None
        min_in_path_y = float('inf')

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            u = (x1 + x2) / 2.0
            v = y2

            X_3d, Y_3d = self.project_2d_to_3d_ground(u, v)

            if Y_3d <= 0 or Y_3d > 100.0:
                continue

            in_path = self.is_inside_lane_roi(X_3d, Y_3d, lane_proposals)

            if not in_path:
                status = "OUT_OF_PATH"
                color = (180, 180, 180) # Gray
            elif Y_3d <= self.danger_dist:
                status = "DANGER"
                color = (0, 0, 255) # Red
            elif Y_3d <= self.warning_dist:
                status = "WARNING"
                color = (0, 255, 255) # Yellow
            else:
                status = "SAFE"
                color = (0, 255, 0) # Green

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
                'is_cipo': False
            }

            if in_path and Y_3d < min_in_path_y:
                min_in_path_y = Y_3d
                cipo_obj = obj_info

            processed_objects.append(obj_info)

        if cipo_obj is not None:
            cipo_obj['is_cipo'] = True

        return processed_objects, cipo_obj
