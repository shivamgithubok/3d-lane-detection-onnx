"""
Qt Quick 3D BEV viewport.

Embeds a View3D inside the existing QWidget cockpit via QQuickWidget.
Same public API as BEVWidget so main_window can swap backends.

Geometry is built in the *lane frame* (see src/tracking/lane_frame.py): the lane
is pinned to the canvas and the ego car moves and yaws within it, which is what
production cockpits do. Everything is sampled uniformly along Y with a constant
point count, so the QML segment pools no longer pop as anchor visibility changes.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from PySide6.QtCore import QUrl, Qt
from PySide6.QtQuickWidgets import QQuickWidget

from src.ui.car_assets import (
    EGO_AUDI,
    KIND_ASSETS,
    KIND_DODGE,
    KIND_MAX,
    KIND_SHC,
    KIND_SKODA,
    KIND_TESLA,
    TRAFFIC_DODGE,
    TRAFFIC_SHC,
    TRAFFIC_SKODA,
    TRAFFIC_TESLA,
    TRUCK_LABELS,
    asset_url,
)
from src.tracking.lane_frame import LaneFrameModel
from src.utils.drivable_area import parse_lane_components

# Local copies — do not import bev_widget (QPainter + sprite pipeline).
BEV_MAX_DIST_M = 60.0
BEV_MAX_LATERAL_M = 14.0
DEFAULT_VIEW_PITCH = 31.0
DEFAULT_VIEW_YAW = 0.0
DEFAULT_ZOOM = 1.05
DEFAULT_CALIB_PITCH = -7.0
DEFAULT_CALIB_H = 1.0

# Uniform sampling of every lane-frame polyline. Constant across frames.
POLY_SAMPLES = 48
CORRIDOR_SEGS = 12
BOUNDARY_SEGS = 12

# Lane-slot presence hysteresis: how a neighbouring marking fades in/out.
SLOT_MATCH_M = 0.90
SLOT_ENTER_HITS = 3
SLOT_EXIT_MISS = 10
MAX_SLOT = 3  # boundaries per side beyond the ego pair

_QML_PATH = os.path.join(os.path.dirname(__file__), "qml", "BevScene.qml")
_YAW_AWAY = 180.0  # same heading as ego (rear toward chase cam)
_SEDAN_CYCLE = (KIND_SKODA, KIND_SHC)


class _SlotTracker:
    """Hysteretic presence for lane boundaries at fixed lane-width multiples.

    Replaces find_outer_lanes(), which re-picked the outermost lane purely by
    mean X every frame and so teleported the road edge by a whole lane width
    whenever a new far marking appeared.
    """

    def __init__(self):
        self._hits = {}
        self._on = {}

    def update(self, present_slots, all_slots):
        for s in all_slots:
            hit = s in present_slots
            h = self._hits.get(s, 0)
            if hit:
                h = min(SLOT_ENTER_HITS, h + 1) if h >= 0 else 1
                self._hits[s] = h
                if h >= SLOT_ENTER_HITS:
                    self._on[s] = SLOT_EXIT_MISS
            else:
                self._hits[s] = 0
                if s in self._on:
                    self._on[s] -= 1
                    if self._on[s] <= 0:
                        del self._on[s]
        return sorted(self._on.keys())

    def reset(self):
        self._hits.clear()
        self._on.clear()


class BevQuick3DWidget(QQuickWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setClearColor(Qt.black)
        self.setAttribute(Qt.WA_AlwaysStackOnTop, False)

        self.proposals = []
        self.processed_objs = []
        self.cipo_status = "SAFE"
        self.left_3d = None
        self.right_3d = None
        self.cinematic_road = True
        self.show_lane_lines = True
        self.lane_frame = LaneFrameModel()
        self._slots = _SlotTracker()
        self._last_traffic_json = None
        self._last_corridor_json = None
        self._last_lane_json = None
        self._last_dash_json = None
        self._last_edge_json = None

        self.setSource(QUrl.fromLocalFile(os.path.abspath(_QML_PATH)))
        if self.status() == QQuickWidget.Error:
            msgs = "; ".join(err.toString() for err in self.errors())
            raise RuntimeError(f"Failed to load BevScene.qml: {msgs}")

        self._push_camera(
            pitch=DEFAULT_VIEW_PITCH,
            yaw=DEFAULT_VIEW_YAW,
            zoom=DEFAULT_ZOOM,
            calib_pitch=DEFAULT_CALIB_PITCH,
            calib_h=DEFAULT_CALIB_H,
            pan_x=0.0,
            pan_y=0.0,
        )
        self._bind_ego_asset()

    def _root(self):
        return self.rootObject()

    def _set(self, name, value):
        root = self._root()
        if root is not None:
            root.setProperty(name, value)

    def _bind_ego_asset(self):
        asset = EGO_AUDI
        if not asset.exists():
            self._set("overlayHint", f"Phase 1 — missing {asset.filename}")
            print(f"[BEV] Ego GLB not found: {asset.path}")
            return
        url = QUrl.fromLocalFile(os.path.abspath(asset.path)).toString()
        self._set("egoGltf", url)
        self._set("egoScale", float(asset.scale))
        self._set("egoRotX", float(asset.rot_x))
        self._set("egoRotY", float(asset.rot_y))
        self._set("egoRotZ", float(asset.rot_z))
        self._set("egoY", float(asset.y))
        self._set("overlayHint", "Phase 4 — CIPO")
        print(f"[BEV] Ego GLB: {asset.path}")
        if TRAFFIC_SKODA.exists():
            self._set("skodaGltf", asset_url(TRAFFIC_SKODA))
            self._set("skodaScale", float(TRAFFIC_SKODA.scale))
            print(f"[BEV] Traffic Skoda: {TRAFFIC_SKODA.path}")
        if TRAFFIC_SHC.exists():
            self._set("shcGltf", asset_url(TRAFFIC_SHC))
            self._set("shcScale", float(TRAFFIC_SHC.scale))
            self._set("shcRotY", float(TRAFFIC_SHC.rot_y))
            print(f"[BEV] Traffic SHC: {TRAFFIC_SHC.path}")
        if TRAFFIC_DODGE.exists():
            self._set("dodgeGltf", asset_url(TRAFFIC_DODGE))
            self._set("dodgeScale", float(TRAFFIC_DODGE.scale))
            print(f"[BEV] Traffic Dodge: {TRAFFIC_DODGE.path}")

    def _push_camera(self, pitch, yaw, zoom, calib_pitch, calib_h, pan_x, pan_y):
        self._set("pitchDeg", float(pitch))
        self._set("yawDeg", float(yaw))
        self._set("zoomFactor", float(zoom))
        self._set("calibPitch", float(calib_pitch))
        self._set("calibH", float(calib_h))
        self._set("panX", float(pan_x))
        self._set("panY", float(pan_y))

    def toggle_cinematic_road(self):
        self.cinematic_road = not self.cinematic_road
        self._set("cinematicRoad", self.cinematic_road)
        return self.cinematic_road

    def toggle_lane_lines(self):
        self.show_lane_lines = not self.show_lane_lines
        self._set("showLaneLines", self.show_lane_lines)
        return self.show_lane_lines

    def set_calibration(self, pitch_deg, height_m):
        self._set("calibPitch", float(pitch_deg))
        self._set("calibH", float(height_m))

    def _bev_visible(self, obj) -> bool:
        z = float(obj.get("Z_3d", 0.0))
        x = float(obj.get("X_3d", 0.0))
        if not (0.5 < z <= BEV_MAX_DIST_M):
            return False
        if abs(x) > BEV_MAX_LATERAL_M:
            return False
        return True

    def _heading_yaw(self, obj) -> float:
        # Highway same-direction traffic: every car faces like ego.
        return _YAW_AWAY

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _poly_segments(ys, xs, width, pal=None, max_segs=BOUNDARY_SEGS):
        """Segment rows from a uniformly sampled lane-frame polyline.

        ys/xs have a fixed length, so the emitted segment count is identical
        every frame and the QML pools never pop in or out.
        """
        n = min(len(ys), len(xs))
        if n < 2:
            return []
        step = max(1, (n - 1) // max_segs)
        segs = []
        i = 0
        while i + step < n:
            x0, y0 = float(xs[i]), float(ys[i])
            x1, y1 = float(xs[i + step]), float(ys[i + step])
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length >= 0.35:
                # Qt yaw: +Y up, length along local Z after yaw. World Δx, Δz=-Δy.
                row = {
                    "x": round(0.5 * (x0 + x1), 2),
                    "z": round(-0.5 * (y0 + y1), 2),
                    "yaw": round(math.degrees(math.atan2(dx, -dy)), 1),
                    "len": round(length, 2),
                    "w": round(float(width), 2),
                }
                if pal is not None:
                    row["c"] = int(pal)
                segs.append(row)
            i += step
        return segs

    def _lane_frame_mean_xs(self, proposals):
        """Mean lateral X of every detected lane, expressed in the lane frame."""
        out = []
        if not proposals:
            return out
        for lane in proposals:
            try:
                xs, ys, _zs, vis = parse_lane_components(lane)
            except Exception:
                continue
            m = vis & (ys > 1.0) & (ys <= BEV_MAX_DIST_M)
            if int(np.sum(m)) < 2:
                continue
            lx = self.lane_frame.to_lane_frame(xs[m], ys[m])
            out.append(float(np.mean(lx)))
        return out

    def _active_slots(self, proposals):
        """Which lane boundaries exist, as signed lane-width multiples.

        Slot k>0 is the k-th boundary right of ego centre, k<0 to the left;
        |k|=1 is the ego lane's own marking.
        """
        half_w = self.lane_frame.half_width
        width = self.lane_frame.lane_width
        candidates = {}
        for k in range(1, MAX_SLOT + 2):
            off = half_w + (k - 1) * width
            candidates[k] = off
            candidates[-k] = -off

        means = self._lane_frame_mean_xs(proposals)
        present = set()
        for mx in means:
            best, best_d = None, SLOT_MATCH_M
            for k, off in candidates.items():
                d = abs(mx - off)
                if d < best_d:
                    best, best_d = k, d
            if best is not None:
                present.add(best)
        # Ego pair is implied by a valid corridor even if the paint is faint.
        if self.lane_frame.valid:
            present.update({-1, 1})
        return self._slots.update(present, list(candidates.keys())), candidates

    # -------------------------------------------------------------- payloads
    def _corridor_payload(self):
        if not self.lane_frame.valid:
            return []
        ys, cx = self.lane_frame.centerline(BEV_MAX_DIST_M, POLY_SAMPLES)
        w = max(0.8, self.lane_frame.lane_width)
        rows = self._poly_segments(ys, cx, w, max_segs=CORRIDOR_SEGS)
        return rows[:14]

    def _dash_payload(self):
        """White ego-lane dashes that scroll backwards with integrated odometry."""
        if not self.lane_frame.valid:
            return []
        half_w = self.lane_frame.half_width
        rows = []
        for span_y0, span_y1 in self.lane_frame.dash_spans(BEV_MAX_DIST_M):
            if span_y1 - span_y0 < 0.45:
                continue
            ys = np.linspace(span_y0, span_y1, 3)
            cx = self.lane_frame.lane_x(ys)
            for sign in (-1.0, 1.0):
                rows.extend(self._poly_segments(ys, cx + sign * half_w, 0.18, max_segs=1))
            if len(rows) >= 28:
                break
        return rows[:28]

    def _edge_payload(self, slots, offsets):
        """Outer road edges from the outermost *stable* boundary on each side."""
        if not self.lane_frame.valid or not slots:
            return []
        left = [k for k in slots if k < 0]
        right = [k for k in slots if k > 0]
        rows = []
        for group, pick in ((left, min), (right, max)):
            if not group:
                continue
            k = pick(group)
            if abs(k) < 2:  # the ego pair is drawn as dashes, not a road edge
                continue
            ys, xs = self.lane_frame.boundary(offsets[k], BEV_MAX_DIST_M, POLY_SAMPLES)
            rows.extend(self._poly_segments(ys, xs, 0.22, max_segs=BOUNDARY_SEGS))
        return rows[:24]

    def _lane_payload(self, slots, offsets):
        """Adjacent lane markings between the ego pair and the road edge."""
        if not self.lane_frame.valid or not slots:
            return []
        left = [k for k in slots if k < 0]
        right = [k for k in slots if k > 0]
        rows = []
        for group, outer in ((left, min(left) if left else None),
                             (right, max(right) if right else None)):
            for k in group:
                if abs(k) < 2 or k == outer:
                    continue  # ego pair -> dashes, outermost -> road edge
                pal = 1 if k < 0 else 2
                ys, xs = self.lane_frame.boundary(offsets[k], BEV_MAX_DIST_M, POLY_SAMPLES)
                rows.extend(self._poly_segments(ys, xs, 0.07, pal=pal, max_segs=BOUNDARY_SEGS))
                if len(rows) >= 36:
                    break
        return rows[:36]

    def _traffic_payload(self, objs):
        rows = []
        if not objs:
            return rows
        visible = [o for o in objs if self._bev_visible(o)]
        visible.sort(key=lambda o: float(o.get("Z_3d", 99.0)))
        cipo = self._pick_cipo(visible)
        cipo_tid = int(cipo.get("track_id", -999)) if cipo is not None else None
        if cipo is not None:
            visible.sort(key=lambda o: (int(o.get("track_id", -1)) != cipo_tid, float(o.get("Z_3d", 99.0))))
        used = {k: 0 for k in KIND_MAX}
        for obj in visible:
            label = str(obj.get("label", "car")).lower()
            truck = any(k in label for k in TRUCK_LABELS)
            if truck and TRAFFIC_DODGE.exists():
                kind = KIND_DODGE
            else:
                tid = int(obj.get("track_id", 0))
                kind = _SEDAN_CYCLE[tid % len(_SEDAN_CYCLE)]
                if not KIND_ASSETS[kind].exists():
                    kind = KIND_SKODA
            if used[kind] >= KIND_MAX[kind]:
                if used[KIND_SKODA] < KIND_MAX[KIND_SKODA] and TRAFFIC_SKODA.exists():
                    kind = KIND_SKODA
                else:
                    continue
            asset = KIND_ASSETS[kind]
            if not asset.exists():
                continue
            z = float(obj["Z_3d"])
            # Traffic lives in the same lane frame as the road, otherwise cars
            # would slide sideways whenever the ego drifted in its lane.
            x = self._traffic_x(float(obj["X_3d"]), z)
            too_close = False
            for prev in rows:
                dx = x - float(prev["posX"])
                dz = z + float(prev["posZ"])  # posZ is -Z
                if dx * dx + dz * dz < 6.25:  # 2.5 m
                    too_close = True
                    break
            if too_close and int(obj.get("track_id", -1)) != cipo_tid:
                continue
            used[kind] += 1
            rows.append({
                "posX": round(x, 2),
                "posY": round(float(asset.y), 2),
                "posZ": round(-z, 2),
                "yawDeg": 180.0,
                "kind": kind,
            })
        return rows

    def _traffic_x(self, x, z):
        if not self.lane_frame.valid:
            return x
        return self.lane_frame.point_to_lane_frame(x, z)

    def _pick_cipo(self, objs):
        vis = [o for o in objs if self._bev_visible(o)]
        if not vis:
            return None
        marked = [o for o in vis if o.get("is_cipo")]
        if marked:
            return min(marked, key=lambda o: float(o.get("Z_3d", 99.0)))
        path = [o for o in vis if o.get("in_path")]
        if path:
            return min(path, key=lambda o: float(o.get("Z_3d", 99.0)))
        return None

    def _push_cipo(self, objs, status):
        cipo = self._pick_cipo(objs or [])
        self._set("cipoStatus", str(status or "SAFE"))
        if cipo is None:
            self._set("cipoVisible", False)
            self._set("cipoDist", 0.0)
            return
        z = float(cipo.get("Z_3d", 0.0))
        self._set("cipoVisible", True)
        self._set("cipoX", round(self._traffic_x(float(cipo.get("X_3d", 0.0)), z), 2))
        self._set("cipoZ", round(-z, 2))
        self._set("cipoDist", round(z, 1))

    def _push_ego_pose(self):
        offset, yaw = self.lane_frame.ego_pose()
        # Qt yaw matches the segment convention atan2(dx, -dy): heading right
        # decreases the angle from the 180° "facing away" base, so negate.
        self._set("egoX", round(float(offset), 3))
        self._set("egoYawDeg", round(-math.degrees(yaw), 2))
        self._set("laneValid", bool(self.lane_frame.valid))
        self._set("laneHeld", bool(self.lane_frame.held))

    def update_bev_data(
        self,
        proposals,
        processed_objs=None,
        cipo_status="SAFE",
        left_3d=None,
        right_3d=None,
        speed_mps=None,
        dt=1.0 / 30.0,
    ):
        self.proposals = proposals if proposals is not None else []
        self.processed_objs = processed_objs if processed_objs is not None else []
        self.cipo_status = cipo_status
        self.left_3d = left_3d
        self.right_3d = right_3d

        self.lane_frame.update(left_3d, right_3d, speed_mps=speed_mps, dt=dt)
        if not self.lane_frame.valid:
            self._slots.reset()
        slots, offsets = self._active_slots(self.proposals)

        self._push_ego_pose()
        self._push_cipo(self.processed_objs, cipo_status)

        payload = json.dumps(self._traffic_payload(self.processed_objs), separators=(",", ":"))
        if payload != self._last_traffic_json:
            self._last_traffic_json = payload
            self._set("trafficJson", payload)
        corr = json.dumps(self._corridor_payload(), separators=(",", ":"))
        if corr != self._last_corridor_json:
            self._last_corridor_json = corr
            self._set("corridorJson", corr)
        lanes = json.dumps(self._lane_payload(slots, offsets), separators=(",", ":"))
        if lanes != self._last_lane_json:
            self._last_lane_json = lanes
            self._set("laneJson", lanes)
        dashes = json.dumps(self._dash_payload(), separators=(",", ":"))
        if dashes != self._last_dash_json:
            self._last_dash_json = dashes
            self._set("dashJson", dashes)
        edges = json.dumps(self._edge_payload(slots, offsets), separators=(",", ":"))
        if edges != self._last_edge_json:
            self._last_edge_json = edges
            self._set("edgeJson", edges)

    def reset_view(self):
        self._push_camera(
            pitch=DEFAULT_VIEW_PITCH,
            yaw=DEFAULT_VIEW_YAW,
            zoom=DEFAULT_ZOOM,
            calib_pitch=DEFAULT_CALIB_PITCH,
            calib_h=DEFAULT_CALIB_H,
            pan_x=0.0,
            pan_y=0.0,
        )


def create_bev_widget(backend: str = "quick3d", parent=None):
    """
    backend: 'quick3d' | 'painter'
    Falls back to QPainter BEVWidget if Quick 3D QML fails to load.
    """
    kind = (backend or "quick3d").strip().lower()
    if kind in ("quick3d", "qtquick3d", "3d"):
        try:
            widget = BevQuick3DWidget(parent)
            print("[BEV] Qt Quick 3D viewport (CIPO)")
            return widget
        except Exception as exc:
            print(f"[BEV] Quick 3D unavailable ({exc}); falling back to QPainter")
    from src.ui.bev_widget import BEVWidget
    print("[BEV] QPainter viewport")
    return BEVWidget(parent)
