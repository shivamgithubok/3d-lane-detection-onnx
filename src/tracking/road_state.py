"""Shared, fail-closed temporal road state for all ADAS entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.inference import lane_filter_config as cfg
from src.tracking.lane_association import LaneTrackerManager
from src.utils.drivable_area import CorridorEMA, EgoLanePairTracker, extract_ego_corridor_3d, pair_gap_m


@dataclass
class RoadState:
    # Immediate model output for visual continuity; never use this for CIPO.
    visual_lanes: list
    # Confirmed/predicted temporal tracks used for road-state decisions.
    lanes: list
    ego_left: Optional[np.ndarray]
    ego_right: Optional[np.ndarray]
    left_corridor_3d: Optional[np.ndarray]
    right_corridor_3d: Optional[np.ndarray]
    status: str
    age_frames: int

    @property
    def has_valid_corridor(self) -> bool:
        return self.left_corridor_3d is not None and self.right_corridor_3d is not None

    @property
    def is_confirmed(self) -> bool:
        return self.status == "CONFIRMED"


class RoadStateEstimator:
    """Tracks lanes and exposes one validated ego-corridor snapshot per frame.

    A predicted corridor is useful to prevent a visual flash during a brief
    miss, but it is deliberately not treated as confirmed for new CIPO
    classification.  Once the bounded track/pair hold expires, the state is
    UNKNOWN and no corridor is emitted.
    """

    def __init__(self) -> None:
        self.lane_tracker = LaneTrackerManager(
            max_missed_frames=cfg.EKF_MAX_MISSED_FRAMES,
            dist_threshold=cfg.EKF_DIST_THRESHOLD_M,
            confirm_hits=cfg.EKF_CONFIRM_HITS,
            require_confirmed=cfg.EKF_REQUIRE_CONFIRMED,
        )
        self.ego_pair_tracker = EgoLanePairTracker()
        self.corridor_ema = CorridorEMA()
        self._age_frames = 0

    def reset(self) -> None:
        self.__init__()

    def update(self, raw_lanes: Optional[Sequence[np.ndarray]], dt: float) -> RoadState:
        raw_lanes = [] if raw_lanes is None else list(raw_lanes)
        tracked_lanes = list(self.lane_tracker.update(raw_lanes, dt=max(1e-3, float(dt))))
        fresh_measurement = bool(raw_lanes)

        ego_left, ego_right = self.ego_pair_tracker.update(tracked_lanes)
        valid_pair = False
        if ego_left is not None and ego_right is not None:
            gap = pair_gap_m(ego_left, ego_right)
            valid_pair = gap is not None and cfg.EGO_LANE_WIDTH_MIN_M <= gap <= cfg.EGO_LANE_WIDTH_MAX_M

        # A held pair is permitted only for the bounded prediction window.
        # It must not be cleared merely because the lane tracker has no output
        # on this frame; that was defeating EgoLanePairTracker's hold policy.
        held_pair = bool(self.ego_pair_tracker.last_meta.get("held", False))
        confirmed_now = bool(tracked_lanes and valid_pair and fresh_measurement and not held_pair)
        if confirmed_now:
            self._age_frames = 0
        else:
            self._age_frames += 1

        can_predict = self._age_frames <= cfg.EGO_PAIR_HOLD_FRAMES
        if valid_pair and can_predict:
            left, right = extract_ego_corridor_3d(
                tracked_lanes, ego_left=ego_left, ego_right=ego_right
            )
            left, right = self.corridor_ema.update(left, right)
        else:
            # Never synthesize a lane from one boundary or a width-invalid pair.
            ego_left = ego_right = None
            left = right = None
            self.corridor_ema.reset()

        if not valid_pair or not can_predict:
            status = "UNKNOWN"
        elif confirmed_now:
            status = "CONFIRMED"
        else:
            status = "PREDICTED"

        return RoadState(
            visual_lanes=tracked_lanes if tracked_lanes else raw_lanes,
            lanes=tracked_lanes,
            ego_left=ego_left,
            ego_right=ego_right,
            left_corridor_3d=left,
            right_corridor_3d=right,
            status=status,
            age_frames=self._age_frames,
        )
