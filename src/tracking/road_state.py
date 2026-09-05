"""Shared, fail-closed temporal road state for all ADAS entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.inference import lane_filter_config as cfg
from src.tracking.lane_association import LaneTrackerManager
from src.utils.drivable_area import (
    CorridorEMA,
    EgoLanePairTracker,
    clip_lane_to_max_y,
    extract_ego_corridor_3d,
    force_corridor_fixed_width,
    lane_assoc_x,
    match_lane_by_x,
    pair_gap_m,
    pair_occupancy_ok,
    reconstruct_opposite_boundary,
    select_onesided_ego_lane,
)


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
    source: str = "none"
    reconstructed_side: Optional[str] = None
    locked_width_m: Optional[float] = None

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

    One-sided reconstruct (P1): if a locked lane width exists and only one ego
    boundary is measured, the missing side is synthesized in the near field
    (Y <= ONESIDED_MAX_Y_M). That corridor is PREDICTED, never CONFIRMED.
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
        self._onesided_age = 0
        self._locked_w: Optional[float] = None
        self._lock_hits = 0
        self._last_left_x: Optional[float] = None
        self._last_right_x: Optional[float] = None

    def reset(self) -> None:
        self.__init__()

    def _lock_width(self, gap: float) -> None:
        g = float(gap)
        if g < cfg.EGO_LANE_WIDTH_MIN_M or g > cfg.EGO_LANE_WIDTH_MAX_M:
            return
        alpha = float(cfg.ONESIDED_W_EMA_ALPHA)
        if self._locked_w is None:
            self._locked_w = g
        else:
            self._locked_w = (1.0 - alpha) * self._locked_w + alpha * g
        self._lock_hits += 1

    def _width_ready(self) -> bool:
        if not bool(getattr(cfg, "ENABLE_ONESIDED_RECONSTRUCT", False)):
            return False
        if self._locked_w is None:
            return False
        if self._lock_hits < int(cfg.ONESIDED_MIN_LOCK_FRAMES):
            return False
        return cfg.EGO_LANE_WIDTH_MIN_M <= self._locked_w <= cfg.EGO_LANE_WIDTH_MAX_M

    def _clear_width_lock(self) -> None:
        self._locked_w = None
        self._lock_hits = 0
        self._last_left_x = None
        self._last_right_x = None
        self._onesided_age = 0

    def _remember_pair_x(self, left, right) -> None:
        mx_l = lane_assoc_x(left)
        mx_r = lane_assoc_x(right)
        if mx_l is not None:
            self._last_left_x = mx_l
        if mx_r is not None:
            self._last_right_x = mx_r

    def _anchor_x(self) -> tuple[Optional[float], Optional[float]]:
        left_x = self._last_left_x
        right_x = self._last_right_x
        meta = self.ego_pair_tracker.last_meta
        if left_x is None:
            left_x = meta.get("left_x")
        if right_x is None:
            right_x = meta.get("right_x")
        return left_x, right_x

    def _raw_pair_present(self, raw_lanes, left_x, right_x) -> bool:
        if not raw_lanes or left_x is None or right_x is None:
            return False
        match_x = float(cfg.ONESIDED_MATCH_X_M)
        left = match_lane_by_x(left_x, raw_lanes, match_x)
        right = match_lane_by_x(right_x, raw_lanes, match_x, exclude=left)
        if left is None or right is None:
            return False
        gap = pair_gap_m(left, right)
        return gap is not None and cfg.EGO_LANE_WIDTH_MIN_M <= gap <= cfg.EGO_LANE_WIDTH_MAX_M

    def _try_onesided(self, raw_lanes, left_x, right_x):
        """Rebuild the missing ego side only from a live detector measurement."""
        if not raw_lanes or self._locked_w is None:
            return False, None, None, None, None, None
        match_x = float(cfg.ONESIDED_MATCH_X_M)
        side, visible = select_onesided_ego_lane(
            raw_lanes, left_x, right_x, match_x_m=match_x
        )
        if side is None or visible is None:
            return False, None, None, None, None, None

        recon = reconstruct_opposite_boundary(visible, side, self._locked_w)
        if recon is None:
            return False, None, None, None, None, None

        vis_clip = clip_lane_to_max_y(visible, cfg.ONESIDED_MAX_Y_M)
        if vis_clip is None:
            vis_clip = visible
        if side == "left":
            ego_left, ego_right = vis_clip, recon
            missing = "right"
        else:
            ego_left, ego_right = recon, vis_clip
            missing = "left"
        mx_l = lane_assoc_x(ego_left)
        mx_r = lane_assoc_x(ego_right)
        if not pair_occupancy_ok(mx_l, mx_r):
            return False, None, None, None, None, None
        mx = lane_assoc_x(visible)
        if mx is not None:
            if side == "left":
                self._last_left_x = mx
                self._last_right_x = mx + self._locked_w
            else:
                self._last_right_x = mx
                self._last_left_x = mx - self._locked_w
        return True, ego_left, ego_right, recon, side, missing

    def update(self, raw_lanes: Optional[Sequence[np.ndarray]], dt: float, speed_mps: Optional[float] = None) -> RoadState:
        raw_lanes = [] if raw_lanes is None else list(raw_lanes)
        tracked_lanes = list(
            self.lane_tracker.update(
                raw_lanes, dt=max(1e-3, float(dt)), speed_mps=speed_mps
            )
        )
        fresh_measurement = bool(raw_lanes)

        ego_left, ego_right = self.ego_pair_tracker.update(tracked_lanes)
        gap = None
        valid_pair = False
        if ego_left is not None and ego_right is not None:
            gap = pair_gap_m(ego_left, ego_right)
            valid_pair = gap is not None and cfg.EGO_LANE_WIDTH_MIN_M <= gap <= cfg.EGO_LANE_WIDTH_MAX_M

        held_pair = bool(self.ego_pair_tracker.last_meta.get("held", False))
        source = str(self.ego_pair_tracker.last_meta.get("source") or "none")
        left_x, right_x = self._anchor_x()

        use_onesided = False
        recon = None
        reconstructed_side = None
        both_raw = self._raw_pair_present(raw_lanes, left_x, right_x)

        if self._width_ready() and not both_raw:
            ok, os_left, os_right, os_recon, _visible_side, missing = self._try_onesided(
                raw_lanes, left_x, right_x
            )
            if ok:
                self._onesided_age += 1
                if self._onesided_age <= int(cfg.ONESIDED_HOLD_FRAMES):
                    use_onesided = True
                    ego_left, ego_right = os_left, os_right
                    recon = os_recon
                    reconstructed_side = missing
                    valid_pair = True
                    source = f"onesided_{'left' if missing == 'right' else 'right'}"
                else:
                    use_onesided = False
        elif both_raw or (valid_pair and not held_pair):
            self._onesided_age = 0

        confirmed_now = bool(
            tracked_lanes
            and valid_pair
            and fresh_measurement
            and not held_pair
            and not use_onesided
        )
        if confirmed_now:
            self._age_frames = 0
            self._onesided_age = 0
            if gap is not None:
                self._lock_width(gap)
            self._remember_pair_x(ego_left, ego_right)
        else:
            self._age_frames += 1
            stale_limit = int(cfg.EGO_PAIR_HOLD_FRAMES) + int(cfg.ONESIDED_HOLD_FRAMES)
            if not use_onesided and self._age_frames > stale_limit:
                self._clear_width_lock()

        can_emit = False
        if use_onesided:
            can_emit = True
        elif valid_pair and self._age_frames <= cfg.EGO_PAIR_HOLD_FRAMES:
            can_emit = True

        left = right = None
        if can_emit and ego_left is not None and ego_right is not None:
            left, right = extract_ego_corridor_3d(
                tracked_lanes,
                ego_left=ego_left,
                ego_right=ego_right,
                lane_width_m=self._locked_w,
            )
            left, right = self.corridor_ema.update(left, right)
            # Keep fill width fixed (target or locked) with constant margin —
            # EKF / held / onesided PREDICTED must not balloon the corridor.
            if bool(getattr(cfg, "CORRIDOR_FORCE_FIXED_WIDTH", True)):
                paint_w = self._locked_w
                if paint_w is None:
                    paint_w = float(cfg.CORRIDOR_WIDTH_CLAMP_M)
                left, right = force_corridor_fixed_width(
                    left, right, width_m=paint_w
                )
            elif (
                use_onesided
                and left is not None
                and right is not None
                and self._locked_w is not None
            ):
                half = 0.5 * float(self._locked_w)
                center_x = 0.5 * (left[:, 0] + right[:, 0])
                left = left.copy()
                right = right.copy()
                left[:, 0] = center_x - half
                right[:, 0] = center_x + half
        else:
            ego_left = ego_right = None
            left = right = None
            self.corridor_ema.reset()

        if not can_emit:
            status = "UNKNOWN"
            source = "none"
            reconstructed_side = None
        elif confirmed_now:
            status = "CONFIRMED"
        else:
            status = "PREDICTED"

        visual_lanes = list(tracked_lanes) if tracked_lanes else list(raw_lanes)
        if use_onesided and recon is not None:
            visual_lanes.append(recon)

        return RoadState(
            visual_lanes=visual_lanes,
            lanes=tracked_lanes,
            ego_left=ego_left,
            ego_right=ego_right,
            left_corridor_3d=left,
            right_corridor_3d=right,
            status=status,
            age_frames=self._age_frames,
            source=source,
            reconstructed_side=reconstructed_side,
            locked_width_m=self._locked_w,
        )
