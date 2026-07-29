import numpy as np
from src.tracking.ekf_lane_tracker import EKFLaneTracker
from src.inference.postprocess import ANCHOR_Y_STEPS

class LaneTrackerManager:
    """
    Multi-Lane EKF Tracking Manager.
    Associates raw Anchor3DLane detections with active EKF tracks across frames.
    """
    def __init__(self, max_missed_frames=10, dist_threshold=2.5):
        self.trackers = []
        self.next_track_id = 1
        self.max_missed_frames = max_missed_frames
        self.dist_threshold = dist_threshold

    def _compute_distance(self, tracker_pts, proposal_pts):
        """Compute mean BEV lateral distance between predicted tracker points and raw proposal points."""
        # tracker_pts: N x 3
        # proposal_pts: N x 3
        # Compare lateral X coordinates across Y steps
        diff_x = np.abs(tracker_pts[:, 0] - proposal_pts[:, 0])
        return np.mean(diff_x)

    def update(self, detected_proposals, dt=0.033):
        """
        Update all EKF trackers with new detected proposals.
        detected_proposals: list of 3D lane point arrays (each array is N x 3)
        Returns: list of confirmed smoothed 3D lane proposals
        """
        # 1. Predict state for all active trackers
        for trk in self.trackers:
            trk.predict(dt=dt)

        if detected_proposals is None or len(detected_proposals) == 0:
            proposal_pts_list = []
        else:
            proposal_pts_list = []
            for prop in detected_proposals:
                if isinstance(prop, np.ndarray) and prop.ndim == 1:
                    xs = prop[5 : 5 + len(ANCHOR_Y_STEPS)]
                    zs = prop[5 + len(ANCHOR_Y_STEPS) : 5 + 2 * len(ANCHOR_Y_STEPS)]
                    ys = ANCHOR_Y_STEPS
                    pts_3d = np.column_stack((xs, ys, zs))
                else:
                    pts_3d = prop
                proposal_pts_list.append(pts_3d)

        num_trackers = len(self.trackers)
        num_proposals = len(proposal_pts_list)

        matched_trackers = set()
        matched_proposals = set()

        if num_trackers > 0 and num_proposals > 0:
            # Build distance cost matrix
            cost_matrix = np.zeros((num_trackers, num_proposals), dtype=np.float32)
            for i, trk in enumerate(self.trackers):
                trk_pts = trk.get_lane_points(ANCHOR_Y_STEPS)
                for j, prop_pts in enumerate(proposal_pts_list):
                    cost_matrix[i, j] = self._compute_distance(trk_pts, prop_pts)

            # Greedy matching (minimum distance association)
            for _ in range(min(num_trackers, num_proposals)):
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_dist = cost_matrix[min_idx]

                if min_dist > self.dist_threshold:
                    break

                trk_idx, prop_idx = min_idx
                if trk_idx not in matched_trackers and prop_idx not in matched_proposals:
                    matched_trackers.add(trk_idx)
                    matched_proposals.add(prop_idx)
                    # Update tracker with matched proposal 3D points
                    self.trackers[trk_idx].update(proposal_pts_list[prop_idx])

                # Prevent re-matching
                cost_matrix[trk_idx, :] = 1e6
                cost_matrix[:, prop_idx] = 1e6

        # 2. Spawn new trackers for unmatched proposals
        for j in range(num_proposals):
            if j not in matched_proposals:
                new_trk = EKFLaneTracker(proposal_pts_list[j], track_id=self.next_track_id)
                self.next_track_id += 1
                self.trackers.append(new_trk)

        # 3. Clean up dead trackers (missing for too long)
        self.trackers = [trk for trk in self.trackers if trk.misses <= self.max_missed_frames]

        # 4. Extract smoothed 3D points from active confirmed trackers
        smoothed_lanes = []
        for trk in self.trackers:
            if trk.is_confirmed or trk.hits >= 2:
                smoothed_lanes.append(trk.get_lane_points(ANCHOR_Y_STEPS))

        return smoothed_lanes
