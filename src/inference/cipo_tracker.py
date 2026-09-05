import numpy as np
from src.utils.drivable_area import find_ego_lanes, parse_lane_components, STANDARD_LANE_WIDTH, pair_gap_m
from src.inference.postprocess import decode_lane_pixels
from src.utils.calibration import P_final
from src.inference import lane_filter_config as cfg

# Single source of truth: cam_height=1.5m, pitch=-3°, then crop/resize to 480x360
# (same as src.utils.calibration.P_final — do not hardcode a divergent copy)
DEFAULT_P_MATRIX = np.asarray(P_final, dtype=np.float64)

ANCHOR_LEN = 20
ANCHOR_Y_STEPS = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], dtype=np.float64)

class CIPOTracker:
    def __init__(self, P_matrix=DEFAULT_P_MATRIX, danger_dist=15.0, warning_dist=30.0, ema_alpha=0.35):
        self.P = P_matrix
        self.danger_dist = danger_dist  # < 15m DANGER (Red)
        self.warning_dist = warning_dist # 15m - 30m WARNING (Yellow)
        self.ema_alpha = ema_alpha       # Temporal Exponential Moving Average smoothing factor
        self.track_history = {}          # History dict for temporal smoothing per track_id
        self._hist_frame = 0
        self._hist_ttl = 20              # drop unused IDs (~1–2s) so ByteTrack reuse doesn't teleport
        # P1: per-track in-path hysteresis (enter/exit)
        self._inpath_state = {}         # tid -> {in_path, hits, miss}

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

    def _lane_aware_x(self, u_model, v_model, y_fwd, x_geom, proposals):
        """
        Snap lateral X to a detected-lane center.

        Using only the ego pair collapses every adjacent car into one slot
        (they overlap on the BEV). All lane lines at this depth are used.
        """
        if proposals is None or len(proposals) == 0:
            return x_geom

        samples = []
        for lane in proposals:
            u = self.get_2d_lane_u_at_v(lane, v_model)
            x = self.get_lane_x_at_y(lane, y_fwd)
            if u is None and x is None:
                continue
            samples.append((u, x))

        xs = sorted(x for _, x in samples if x is not None)
        uniq = []
        for x in xs:
            if not uniq or abs(x - uniq[-1]) > 0.8:
                uniq.append(x)
        xs = uniq
        if not xs:
            return x_geom

        if len(xs) == 1:
            w = STANDARD_LANE_WIDTH
            centers = [xs[0] - w, xs[0], xs[0] + w]
        else:
            widths = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            med_w = float(np.median(widths))
            med_w = max(2.8, min(4.6, med_w))
            centers = [xs[0] - 0.5 * med_w]
            for i in range(len(xs) - 1):
                centers.append(0.5 * (xs[i] + xs[i + 1]))
            centers.append(xs[-1] + 0.5 * med_w)

        with_u = [(u, x) for u, x in samples if u is not None and x is not None]
        with_u.sort(key=lambda t: t[0])
        snapped = None
        if len(with_u) >= 2:
            us = [t[0] for t in with_u]
            xsu = [t[1] for t in with_u]
            if u_model <= us[0]:
                snapped = centers[0]
            elif u_model >= us[-1]:
                snapped = centers[-1]
            else:
                for i in range(len(us) - 1):
                    if us[i] <= u_model <= us[i + 1]:
                        snapped = 0.5 * (xsu[i] + xsu[i + 1])
                        break
        if snapped is None:
            snapped = min(centers, key=lambda c: abs(c - x_geom))

        # Small within-lane offset from geometry, without collapsing lanes
        return snapped + 0.12 * float(np.clip(x_geom - snapped, -1.5, 1.5))

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

    def process_detections(
        self,
        detections,
        lane_proposals,
        frame_size=(1080, 720),
        depth_map=None,
        depth_estimator=None,
        ego_left=None,
        ego_right=None,
        frame_transform=None,
        road_state_confirmed=True,
    ):
        processed_objects = []
        w_img, h_img = frame_size
        scale_u = 480.0 / float(w_img)
        scale_v = 360.0 / float(h_img)

        # A caller with a shared road state supplies the validated ego pair.
        # The legacy fallback is retained only for old callers; production paths
        # pass road_state_confirmed=False when the pair is missing or stale.
        if road_state_confirmed and ego_left is None and ego_right is None:
            ego_left, ego_right = find_ego_lanes(lane_proposals, ANCHOR_LEN)
        self._hist_frame += 1
        seen_ids = set()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det.get('track_id', -1)
            # Ground contact: bottom-center of the 2D box
            u_img = (x1 + x2) / 2.0
            v_img = float(y2)

            if frame_transform is not None:
                if not frame_transform.source_point_is_visible_to_model(u_img, v_img):
                    # This object is outside the lane model's calibrated crop;
                    # never classify it as in-path from unrelated coordinates.
                    continue
                u_model, v_model = frame_transform.source_to_model(
                    np.array([[u_img, v_img]], dtype=np.float64)
                )[0]
            else:
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
                u_model, v_model, Y_meas, X_geom, lane_proposals
            )

            # 3. Temporal EMA smoothing per track_id (skip stale IDs ByteTrack may reuse)
            hist = self.track_history.get(track_id) if track_id > 0 else None
            if hist is not None and hist.get('frame', 0) >= self._hist_frame - 2:
                X_3d = self.ema_alpha * X_meas + (1.0 - self.ema_alpha) * hist['X']
                Y_3d = self.ema_alpha * Y_meas + (1.0 - self.ema_alpha) * hist['Z']
            else:
                X_3d, Y_3d = X_meas, Y_meas

            if track_id > 0:
                self.track_history[track_id] = {
                    'X': X_3d, 'Z': Y_3d, 'frame': self._hist_frame
                }
                seen_ids.add(track_id)

            # 4. In-path / CIPO from 2D lane association + lateral X gate (P1)
            u_margin = cfg.CIPO_U_MARGIN_PX
            u_left_2d = self.get_2d_lane_u_at_v(ego_left, v_model) if ego_left is not None else None
            u_right_2d = self.get_2d_lane_u_at_v(ego_right, v_model) if ego_right is not None else None

            raw_in = False
            if not road_state_confirmed:
                # A stale/unknown road model may be rendered as degraded, but
                # must not create a new in-path safety assertion.
                raw_in = False
            elif u_left_2d is not None and u_right_2d is not None:
                raw_in = (u_left_2d - u_margin) <= u_model <= (u_right_2d + u_margin)
            elif u_left_2d is not None:
                raw_in = u_model >= (u_left_2d - u_margin) and abs(X_3d) <= 2.0
            elif u_right_2d is not None:
                raw_in = u_model <= (u_right_2d + u_margin) and abs(X_3d) <= 2.0
            else:
                raw_in = abs(X_3d) <= 1.50

            # Lateral X gate from measured ego width (rejects adjacent-lane FPs)
            if raw_in and ego_left is not None and ego_right is not None:
                gap = pair_gap_m(ego_left, ego_right, ANCHOR_LEN)
                half = (0.5 * gap) if gap is not None else 1.85
                if abs(X_3d) > half + cfg.CIPO_X_MARGIN_M:
                    raw_in = False

            # Hysteresis per track_id (and anonymous boxes keyed by rounded bbox)
            state_key = track_id if track_id > 0 else f"anon_{int(u_img)}_{int(v_img)}"
            st = self._inpath_state.get(state_key)
            if st is None:
                st = {"in_path": False, "hits": 0, "miss": 0}
            if raw_in:
                st["hits"] += 1
                st["miss"] = 0
                if st["hits"] >= cfg.CIPO_ENTER_HITS or st["in_path"]:
                    st["in_path"] = True
            else:
                st["miss"] += 1
                st["hits"] = 0
                if st["miss"] >= cfg.CIPO_EXIT_MISS:
                    st["in_path"] = False
            st["last_frame"] = self._hist_frame
            self._inpath_state[state_key] = st
            in_path = bool(st["in_path"])

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
            show_bev = True

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

        stale = [
            tid for tid, h in self.track_history.items()
            if h.get('frame', 0) < self._hist_frame - self._hist_ttl
        ]
        for tid in stale:
            self.track_history.pop(tid, None)
        # Keep hysteresis through short detector misses.  Removing all state
        # whenever one frame has no box defeated the configured exit hysteresis.
        self._inpath_state = {
            k: v for k, v in self._inpath_state.items()
            if int(v.get("last_frame", 0)) >= self._hist_frame - self._hist_ttl
        }

        cipo_obj = None
        if road_state_confirmed:
            in_path = [obj for obj in processed_objects if obj["in_path"]]
            if in_path:
                cipo_obj = min(in_path, key=lambda obj: obj["Z_3d"])
        return processed_objects, cipo_obj
