import numpy as np
from src.utils.drivable_area import find_ego_lanes, parse_lane_components, STANDARD_LANE_WIDTH
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
    def __init__(self, P_matrix=DEFAULT_P_MATRIX, danger_dist=15.0, warning_dist=30.0, ema_alpha=0.35):
        self.P = P_matrix
        self.danger_dist = danger_dist  # < 15m DANGER (Red)
        self.warning_dist = warning_dist # 15m - 30m WARNING (Yellow)
        self.ema_alpha = ema_alpha       # Temporal Exponential Moving Average smoothing factor
        self.track_history = {}          # History dict for temporal smoothing per track_id

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

        Y = max(1.0, min(100.0, Y))

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

        order = np.argsort(vs)
        vs = np.array(vs)[order]
        us = np.array(us)[order]

        if v_target < vs[0] or v_target > vs[-1]:
            return None

        u_interp = float(np.interp(v_target, vs, us))
        return u_interp

    def get_lane_x_at_y(self, lane_proposal, y_target):
        """Interpolate 3D lateral X of a lane line at forward distance y_target (meters)."""
        if lane_proposal is None:
            return None
        xs, ys, zs, vis = parse_lane_components(lane_proposal, ANCHOR_LEN)
        if vis.sum() < 2:
            return None
        order = np.argsort(ys[vis])
        y_valid = ys[vis][order]
        x_valid = xs[vis][order]
        y_t = float(np.clip(y_target, y_valid[0], y_valid[-1]))
        return float(np.interp(y_t, y_valid, x_valid))

    def _lane_aware_x(self, u_model, v_model, y_fwd, x_geom, ego_left, ego_right):
        """
        Place the car in the correct BEV lane using 2D image-lane association,
        then map onto 3D lane X at the vehicle's forward distance.

        Geometric/depth X alone underestimates |X| — adjacent-lane cars collapse
        into the ego corridor on the BEV. Lane association is the reliable signal.
        """
        u_left = self.get_2d_lane_u_at_v(ego_left, v_model) if ego_left is not None else None
        u_right = self.get_2d_lane_u_at_v(ego_right, v_model) if ego_right is not None else None

        x_left = self.get_lane_x_at_y(ego_left, y_fwd)
        x_right = self.get_lane_x_at_y(ego_right, y_fwd)

        # Ego corridor width from detected boundaries (fallback = standard lane)
        if x_left is not None and x_right is not None:
            ego_width = max(2.5, abs(x_right - x_left))
        else:
            ego_width = STANDARD_LANE_WIDTH

        # ── Left of ego-left boundary → left adjacent lane ──
        if u_left is not None and u_model < u_left - 3.0:
            if x_left is not None:
                # Map how far left of the boundary (image px) into lane widths
                # ~80 model-px ≈ one full adjacent lane
                px_left = max(0.0, (u_left - 3.0) - u_model)
                t = float(np.clip(px_left / 80.0, 0.30, 1.20))
                x_lane = x_left - t * ego_width
            else:
                x_lane = -ego_width * 1.0
            # Keep mild geometric spread when already clearly left
            if x_geom < x_lane:
                return 0.75 * x_lane + 0.25 * x_geom
            return x_lane

        # ── Right of ego-right boundary → right adjacent lane ──
        if u_right is not None and u_model > u_right + 3.0:
            if x_right is not None:
                px_right = max(0.0, u_model - (u_right + 3.0))
                t = float(np.clip(px_right / 80.0, 0.30, 1.20))
                x_lane = x_right + t * ego_width
            else:
                x_lane = ego_width * 1.0
            if x_geom > x_lane:
                return 0.75 * x_lane + 0.25 * x_geom
            return x_lane

        # ── Inside ego lane in the camera → interpolate between ego boundaries ──
        if u_left is not None and u_right is not None and x_left is not None and x_right is not None:
            denom = (u_right - u_left)
            if abs(denom) > 1e-3:
                t = float(np.clip((u_model - u_left) / denom, 0.05, 0.95))
                return x_left + t * (x_right - x_left)

        # Fallback: geometric / depth X
        return x_geom

    def _lane_rank(self, u_model, v_model, x_3d, ego_left, ego_right):
        """
        0 = ego lane, 1 = adjacent (2nd) lane, 2+ = outer / 3rd+ lanes.

        Outer lanes are not drawn on the BEV car overlay.
        """
        u_left = self.get_2d_lane_u_at_v(ego_left, v_model) if ego_left is not None else None
        u_right = self.get_2d_lane_u_at_v(ego_right, v_model) if ego_right is not None else None
        x_left = self.get_lane_x_at_y(ego_left, 25.0)
        x_right = self.get_lane_x_at_y(ego_right, 25.0)
        if x_left is not None and x_right is not None:
            ego_half = 0.5 * abs(x_right - x_left)
            lane_w = max(STANDARD_LANE_WIDTH, abs(x_right - x_left))
        else:
            ego_half = STANDARD_LANE_WIDTH * 0.5
            lane_w = STANDARD_LANE_WIDTH

        # ~80 model-px ≈ one adjacent lane in image space
        adj_px = 85.0

        if u_left is not None and u_model < u_left - 3.0:
            px = (u_left - 3.0) - u_model
            return 1 if px <= adj_px else 2
        if u_right is not None and u_model > u_right + 3.0:
            px = u_model - (u_right + 3.0)
            return 1 if px <= adj_px else 2

        # Inside ego corridor in image, or unknown 2D → refine with X
        if abs(x_3d) <= ego_half + 0.6:
            return 0
        if abs(x_3d) <= ego_half + lane_w * 1.15:
            return 1
        return 2

    def process_detections(self, detections, lane_proposals, frame_size=(1080, 720), depth_map=None, depth_estimator=None):
        processed_objects = []
        w_img, h_img = frame_size
        scale_u = 480.0 / float(w_img)
        scale_v = 360.0 / float(h_img)

        ego_left, ego_right = find_ego_lanes(lane_proposals, ANCHOR_LEN)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det.get('track_id', -1)
            # Ground contact: bottom-center of the 2D box
            u_img = (x1 + x2) / 2.0
            v_img = float(y2)

            u_model = u_img * scale_u
            v_model = v_img * scale_v

            # 1. Forward distance (depth) + geometric lateral seed
            if depth_map is not None and depth_estimator is not None:
                Y_meas = depth_estimator.query_vehicle_depth(depth_map, det['bbox'], w_img, h_img)
                denom_x = self.P[0, 0]
                X_geom = float((Y_meas * (self.P[2, 1] * u_model - self.P[0, 1])) / denom_x)
            else:
                X_geom, Y_meas = self.project_2d_to_3d_ground(u_model, v_model)

            if Y_meas <= 0 or Y_meas > 100.0:
                continue

            # 2. Lane-aware lateral X (keeps adjacent-lane cars out of ego corridor)
            X_meas = self._lane_aware_x(
                u_model, v_model, Y_meas, X_geom, ego_left, ego_right
            )

            # 3. Temporal EMA smoothing per track_id
            if track_id > 0 and track_id in self.track_history:
                hist = self.track_history[track_id]
                X_3d = self.ema_alpha * X_meas + (1.0 - self.ema_alpha) * hist['X']
                Y_3d = self.ema_alpha * Y_meas + (1.0 - self.ema_alpha) * hist['Z']
            else:
                X_3d, Y_3d = X_meas, Y_meas

            if track_id > 0:
                self.track_history[track_id] = {'X': X_3d, 'Z': Y_3d}

            # 4. In-path / CIPO from 2D lane association (camera space)
            u_left_2d = self.get_2d_lane_u_at_v(ego_left, v_model) if ego_left is not None else None
            u_right_2d = self.get_2d_lane_u_at_v(ego_right, v_model) if ego_right is not None else None

            is_left_of_left_lane = (u_left_2d is not None) and (u_model < u_left_2d - 3.0)
            is_right_of_right_lane = (u_right_2d is not None) and (u_model > u_right_2d + 3.0)

            if is_left_of_left_lane or is_right_of_right_lane:
                in_path = False
            elif u_left_2d is not None and u_right_2d is not None:
                in_path = (u_left_2d - 3.0) <= u_model <= (u_right_2d + 3.0)
            else:
                in_path = abs(X_3d) <= 1.50

            if not in_path:
                status = "OUT OF PATH"
                color = (255, 220, 0)
            elif Y_3d < self.danger_dist:
                status = "DANGER <15m"
                color = (0, 0, 255)
            else:
                status = f"IN PATH ({Y_3d:.1f}m)"
                color = (0, 215, 255)

            Y_ground = float(((v_model * Y_3d) - self.P[1, 2] * Y_3d - self.P[1, 3]) / (self.P[1, 1] + 1e-6))

            lane_rank = self._lane_rank(u_model, v_model, X_3d, ego_left, ego_right)
            # BEV overlay: ego + adjacent (2nd) only — hide 3rd+ lanes
            show_bev = lane_rank <= 1

            obj_info = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': det.get('class', 'car'),
                'track_id': track_id,
                'conf': det.get('conf', 1.0),
                'X_3d': X_3d,
                'Z_3d': Y_3d,
                'Y_ground': Y_ground,
                'in_path': in_path,
                'status': status,
                'color': color,
                'is_cipo': in_path and Y_3d < self.danger_dist,
                'lane_rank': lane_rank,
                'show_bev': show_bev,
            }

            processed_objects.append(obj_info)

        return processed_objects, None
