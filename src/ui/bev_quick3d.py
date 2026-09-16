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
import time

import numpy as np

from PySide6.QtCore import QUrl, Qt, QTimer
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
from src.tracking.lane_assignment import LaneAssigner, LaneModel, place_in_lane
from src.tracking.heading import CROSS_MODES, HeadingEstimator
from src.tracking.ego_pose import EgoPose
from src.utils.drivable_area import parse_lane_components

# Local copies — do not import bev_widget (QPainter + sprite pipeline).
BEV_MAX_DIST_M = 70.0
BEV_DISK_R_M = 70.0
BEV_MAX_LATERAL_M = 14.0
TURN_YAW_MAX_DEG = 14.0
TURN_YAW_GAIN = 0.45
TURN_YAW_ALPHA = 0.22
DEFAULT_VIEW_PITCH = 13.0
DEFAULT_VIEW_YAW = 0.0
DEFAULT_ZOOM = 1.08
DEFAULT_CALIB_PITCH = 0.0
DEFAULT_CALIB_H = 1.0

# Layer A: ego body keep-out (half sedan length ~2.3 m + bumper margin).
EGO_KEEP_OUT_M = 5.0
# Render-tick coast between ~12 Hz inference frames.
MAX_EXTRAPOLATE_S = 0.20

TRAFFIC_LANE_MAX = 1          # BEV: ego + one each side; rest stay camera-only
DEFAULT_LANE_W_M = 3.7
SAME_LANE_GAP_M = 8.0         # min range gap for two cars in the same lane

# Uniform sampling of every lane-frame polyline. Constant across frames.
POLY_SAMPLES = 48
CORRIDOR_SEGS = 16
BOUNDARY_SEGS = 12
CORRIDOR_START_M = 5.0   # start ahead of the ego body, not under the car
CORRIDOR_DRAW_M = 30.0   # practical cluster lookahead

# Lane-slot presence hysteresis: how a neighbouring marking fades in/out.
SLOT_MATCH_M = 0.90
SLOT_ENTER_HITS = 3
SLOT_EXIT_MISS = 10
MAX_SLOT = 3  # boundaries per side beyond the ego pair

_QML_PATH = os.path.join(os.path.dirname(__file__), "qml", "BevScene.qml")
_ENV_MODES = ("auto", "day", "dusk", "night", "demo")


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
        # Transparent so the QML 2D sky gradient behind View3D is visible.
        self.setClearColor(Qt.transparent)
        self.setAttribute(Qt.WA_AlwaysStackOnTop, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.proposals = []
        self.processed_objs = []
        self.cipo_status = "SAFE"
        self.left_3d = None
        self.right_3d = None
        self.cinematic_road = True
        self.show_lane_lines = True
        self.env_mode = "auto"
        self.lane_frame = LaneFrameModel()
        self._slots = _SlotTracker()
        self._tid_slot = {}   # track_id → (mesh_kind, mesh_pool_index)
        self._lane_assign = LaneAssigner(max_index=TRAFFIC_LANE_MAX)
        self._heading = HeadingEstimator()
        self._kind_vote = {}  # tid -> {truck, hits}
        self._ego_pose = EgoPose()
        self._turn_yaw_deg = 0.0
        self._world_flash_deg = 0.0
        self._ego_speed_mps = None
        self._last_traffic_json = None
        self._last_corridor_json = None
        self._last_lane_json = None
        self._last_dash_json = None
        self._last_edge_json = None
        self._traffic_seed = []
        self._traffic_t0 = 0.0

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
        self._set("envMode", self.env_mode)
        self._set("diskRadiusM", float(BEV_DISK_R_M))
        self._set("debugMaxZ", float(BEV_MAX_DIST_M))
        self._set("trailJson", "[]")
        self._bind_ego_asset()
        self._extrap = QTimer(self)
        self._extrap.setInterval(33)
        self._extrap.timeout.connect(self._on_extrap_tick)
        self._extrap.start()

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
            print(f"[BEV] Traffic truck: {TRAFFIC_DODGE.path}")
        else:
            print(f"[BEV] Traffic truck missing: {TRAFFIC_DODGE.path}")

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

    def toggle_scenic_view(self):
        """Full scene (sky/grass/mountains) vs one-color road-only semantic."""
        root = self._root()
        cur = True
        if root is not None:
            val = root.property("scenicView")
            if val is not None:
                cur = bool(val)
        nxt = not cur
        self._set("scenicView", nxt)
        return nxt

    def cycle_env_mode(self):
        """Cycle BEV background: Auto → Day → Dusk → Night → Demo."""
        root = self._root()
        if root is not None:
            cur = root.property("envMode")
            if cur:
                self.env_mode = str(cur)
        try:
            i = _ENV_MODES.index(self.env_mode)
        except ValueError:
            i = -1
        self.env_mode = _ENV_MODES[(i + 1) % len(_ENV_MODES)]
        self._set("envMode", self.env_mode)
        return self.env_mode

    def set_env_mode(self, mode: str):
        mode = str(mode).lower().strip()
        if mode not in _ENV_MODES:
            raise ValueError(f"env mode must be one of {_ENV_MODES}")
        self.env_mode = mode
        self._set("envMode", self.env_mode)
        return self.env_mode

    def set_calibration(self, pitch_deg, height_m):
        self._set("calibPitch", float(pitch_deg))
        self._set("calibH", float(height_m))

    def toggle_debug_metric(self):
        """Ego centreline at x=0, 10 m ruler ticks, per-object (x, z) in metres.

        Reads the same trafficJson the renderer consumes, so the printed numbers
        cannot drift from the drawn positions. Pair with
        scripts/debug/range_diagnostics.py to check BEV placement against the
        per-frame CSV.
        """
        root = self._root()
        cur = bool(root.property("debugMetric")) if root is not None else False
        self._set("debugMetric", not cur)
        self._set("debugMaxZ", float(BEV_MAX_DIST_M))
        return not cur

    def _bev_visible(self, obj) -> bool:
        z = float(obj.get("Z_3d", 0.0))
        x = float(obj.get("X_3d", 0.0))
        # Layer A: never draw into the ego GLB body.
        if not (EGO_KEEP_OUT_M < z <= BEV_MAX_DIST_M):
            return False
        if math.hypot(x, z) > BEV_DISK_R_M:
            return False
        # Lateral hide uses image lane index, not metric X (X is a different camera).
        if int(obj.get("lane_rank", 0)) > TRAFFIC_LANE_MAX:
            return False
        if abs(int(obj.get("lane_index", 0))) > TRAFFIC_LANE_MAX:
            return False
        if abs(x) > BEV_MAX_LATERAL_M and obj.get("lane_index") is None:
            return False
        return True

    def _lane_width_m(self) -> float:
        if self.lane_frame.valid:
            return max(2.8, min(4.6, float(self.lane_frame.lane_width)))
        return DEFAULT_LANE_W_M

    def _measured_lane_slot(self, obj) -> int:
        """Lane index at the object's own range. Never force in-path to slot 0."""
        z = max(1.0, float(obj.get("Z_3d", 1.0)))
        x_cam = float(obj.get("X_3d", 0.0))
        x_lf = self._traffic_x(x_cam, z)
        w = self._lane_width_m()
        raw = int(round(x_lf / w))
        return int(np.clip(raw, -TRAFFIC_LANE_MAX, TRAFFIC_LANE_MAX))

    def _heading_yaw(self, obj, y, lane_model):
        """Lane tangent, or world-velocity class when ego speed is known."""
        return self._heading.update(
            int(obj.get("track_id", -1)),
            float(obj.get("vx", 0.0)),
            float(obj.get("vy", 0.0)),
            float(y),
            self._ego_speed_mps,
            lane_model,
        )

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _poly_segments(ys, xs, width, pal=None, max_segs=BOUNDARY_SEGS, radius=BEV_DISK_R_M,
                       y_min=0.0):
        """Segment rows from a uniformly sampled lane-frame polyline.

        ys/xs have a fixed length, so the emitted segment count is identical
        every frame and the QML pools never pop in or out. Live paint stays in
        the forward 70 m disk; sides and rear stay empty.
        """
        ys = np.asarray(ys, dtype=np.float64)
        xs = np.asarray(xs, dtype=np.float64)
        n = min(len(ys), len(xs))
        if n < 2:
            return []
        ys, xs = ys[:n], xs[:n]
        if radius is not None:
            r2 = float(radius) * float(radius)
            keep = (ys >= float(y_min)) & ((xs * xs + ys * ys) <= r2)
            if int(np.sum(keep)) < 2:
                return []
            ys, xs = ys[keep], xs[keep]
            n = len(ys)
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
        """Ego-lane path ahead of the bumper — not a pad under the car."""
        if not self.lane_frame.valid:
            return []
        ys = np.linspace(CORRIDOR_START_M, CORRIDOR_DRAW_M, POLY_SAMPLES)
        xs = self.lane_frame.lane_x(ys)
        w = float(np.clip(self.lane_frame.lane_width * 0.94, 2.6, 3.6))
        rows = self._poly_segments(
            ys, xs, w, max_segs=CORRIDOR_SEGS, y_min=CORRIDOR_START_M,
        )
        return rows[:18]

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
                rows.extend(self._poly_segments(ys, cx + sign * half_w, 0.14, max_segs=1))
            if len(rows) >= 28:
                break
        return rows[:28]

    def _edge_payload(self, slots, offsets):
        """Dashboard BEV shows only the ego lane — no outer road edges."""
        return []

    def _lane_payload(self, slots, offsets):
        """Dashboard BEV shows only the ego lane — no adjacent markings."""
        return []

    def _prefer_kind(self, obj):
        """Truck mesh only after a sticky vote — one YOLO 'truck' frame is not enough."""
        tid = int(obj.get("track_id", -1))
        label = str(obj.get("label", "car")).lower().strip()
        meas = label in TRUCK_LABELS
        st = self._kind_vote.get(tid)
        if st is None:
            st = {"truck": False, "hits": 0}
            self._kind_vote[tid] = st
        if meas == st["truck"]:
            st["hits"] = 0
        else:
            st["hits"] += 1
            need = 5 if meas else 3
            if st["hits"] >= need:
                st["truck"] = meas
                st["hits"] = 0
        if not st["truck"] or not TRAFFIC_DODGE.exists():
            return KIND_SKODA
        bbox = obj.get("bbox") or [0, 0, 1, 1]
        bw = abs(float(bbox[2]) - float(bbox[0]))
        z = max(1.0, float(obj.get("Z_3d", 1.0)))
        # Tiny far box: sedan YOLO-labelled truck. ~1.5 m wide at f≈700.
        if bw * z < 1100.0:
            return KIND_SKODA
        if KIND_MAX.get(KIND_DODGE, 0) <= 0:
            return KIND_SKODA
        return KIND_DODGE

    def _alloc_slot(self, tid, prefer_kind):
        """Keep (kind, slot) sticky per track_id so QML nodes do not swap cars."""
        existing = self._tid_slot.get(tid)
        if existing is not None:
            kind, slot = existing
            # Drop SHC/Tesla leftovers: those meshes were centimetre-scale specks.
            if kind in (KIND_SHC, KIND_TESLA) or KIND_MAX.get(kind, 0) <= slot:
                self._tid_slot.pop(tid, None)
            elif kind != prefer_kind:
                self._tid_slot.pop(tid, None)
            else:
                return existing
        taken = {k: set() for k in KIND_MAX}
        for k, s in self._tid_slot.values():
            taken.setdefault(k, set()).add(s)
        for kind in (prefer_kind, KIND_SKODA, KIND_DODGE):
            cap = int(KIND_MAX.get(kind, 0))
            if cap <= 0 or not KIND_ASSETS[kind].exists():
                continue
            used = taken.get(kind, set())
            for slot in range(cap):
                if slot not in used:
                    self._tid_slot[tid] = (kind, slot)
                    return kind, slot
        return None

    def _traffic_payload(self, objs):
        rows = []
        if not objs:
            self._tid_slot = {}
            self._lane_assign.drop([])
            self._heading.drop([])
            self._kind_vote = {}
            return rows
        visible = [o for o in objs if self._bev_visible(o)]
        live = set()
        for obj in visible:
            tid = int(obj.get("track_id", -1))
            if tid > 0:
                live.add(tid)
        self._tid_slot = {tid: slot for tid, slot in self._tid_slot.items() if tid in live}
        self._lane_assign.drop(live)
        self._heading.drop(live)
        self._kind_vote = {t: s for t, s in self._kind_vote.items() if t in live}

        cipo = self._pick_cipo(visible)
        cipo_tid = int(cipo.get("track_id", -999)) if cipo is not None else None
        lane_model = LaneModel.from_lane_frame(self.lane_frame, n_left=1, n_right=1)
        ordered = sorted(
            visible,
            key=lambda o: (
                int(o.get("track_id", -1)) != cipo_tid,
                int(o.get("track_id", -1)) not in self._tid_slot,
                float(o.get("Z_3d", 99.0)),
            ),
        )
        pending = []
        for obj in ordered:
            tid = int(obj.get("track_id", -1))
            if tid <= 0:
                continue
            z = float(obj["Z_3d"])
            if z <= EGO_KEEP_OUT_M:
                continue

            x_cam = float(obj.get("X_3d", 0.0))
            raw_off = float(obj.get("lane_offset_m", 0.0))
            if "lane_index" in obj:
                raw_idx = int(obj.get("lane_index", 0))
                if abs(raw_idx) > TRAFFIC_LANE_MAX:
                    continue
                lane_slot, off = self._lane_assign.update_index(tid, raw_idx, raw_off)
            else:
                lane_slot, off = self._lane_assign.update(tid, x_cam, z, lane_model)
            if lane_slot is None:
                lane_slot = self._measured_lane_slot(obj)
                off = raw_off
            if abs(int(lane_slot or 0)) > TRAFFIC_LANE_MAX:
                continue
            pending.append((obj, tid, int(lane_slot), z, float(off)))

        pending = self._spread_same_lane(pending)

        for obj, tid, lane_slot, z, off in pending:
            yaw, mode = self._heading_yaw(obj, z, lane_model)
            if mode in CROSS_MODES:
                x = float(self._traffic_x(float(obj.get("X_3d", 0.0)), z))
                y = float(z)
            else:
                w = self._lane_width_m()
                if lane_model is not None:
                    x_meas = float(lane_model.lane_center_x(int(lane_slot), z)) + float(off)
                else:
                    x_meas = float(lane_slot) * w + float(off)
                x, y = place_in_lane(
                    x_meas, z, lane_model, int(lane_slot),
                    snap_strength=0.0, offset_clamp_m=1.15,
                )
                if lane_model is None:
                    x, y = float(x_meas), z

            bound = self._alloc_slot(tid, self._prefer_kind(obj))
            if bound is None:
                continue
            kind, mesh_slot = bound
            asset = KIND_ASSETS[kind]
            rows.append({
                "tid": tid,
                "kind": kind,
                "slot": mesh_slot,
                "lane": int(lane_slot),
                "posX": round(float(x), 2),
                "posY": round(float(asset.y), 2),
                "posZ": round(-float(y), 2),
                "x0": float(x),
                "y0": float(y),
                "vx": float(obj.get("vx", 0.0)),
                "vy": float(obj.get("vy", 0.0)),
                "yawDeg": round(float(yaw), 1),
                "head": mode,
            })
        return rows

    @staticmethod
    def _spread_same_lane(pending):
        """Keep two cars in one lane from stacking: nearer stays, farther slides back."""
        if len(pending) < 2:
            return pending
        by_lane = {}
        for item in pending:
            by_lane.setdefault(item[2], []).append(item)
        out = []
        for _lane, items in by_lane.items():
            items.sort(key=lambda t: t[3])
            last_z = None
            for obj, tid, lane_slot, z, off in items:
                if last_z is not None and z < last_z + SAME_LANE_GAP_M:
                    z = last_z + SAME_LANE_GAP_M
                last_z = z
                out.append((obj, tid, lane_slot, z, off))
        return out

    def _on_extrap_tick(self):
        seed = self._traffic_seed
        if not seed or self._traffic_t0 <= 0.0:
            return
        dt = min(MAX_EXTRAPOLATE_S, time.perf_counter() - self._traffic_t0)
        if dt < 0.012:
            return
        live = []
        for r in seed:
            x = float(r["x0"]) + float(r.get("vx", 0.0)) * dt
            y = float(r["y0"]) + float(r.get("vy", 0.0)) * dt
            out = dict(r)
            out["posX"] = round(x, 2)
            out["posZ"] = round(-y, 2)
            live.append(out)
        payload = json.dumps(live, separators=(",", ":"))
        if payload != self._last_traffic_json:
            self._last_traffic_json = payload
            self._set("trafficJson", payload)

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
        path = [o for o in vis if o.get("in_path") and int(o.get("lane_index", 0)) == 0]
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
        self._set("cipoDist", round(z, 1))
        # Keep HUD distance, but never park the CIPO beacon inside the ego body.
        if z <= EGO_KEEP_OUT_M:
            self._set("cipoVisible", False)
            return
        self._set("cipoVisible", True)
        # CIPO marker sits on ego-lane centre (slot 0) in the lane frame.
        self._set("cipoX", 0.0)
        self._set("cipoZ", round(-z, 2))

    def _push_ego_pose(self):
        offset, _lane_yaw = self.lane_frame.ego_pose()
        # Brief 3rd-person turn: yaw the ego GLB from v·κ, counter-yaw the
        # world a little, then both EMA back to heading-up (0). Do not apply
        # accumulated odometry or the chase cam just orbits.
        rate_deg = math.degrees(self._ego_pose.yaw_rate)
        target = float(np.clip(rate_deg * TURN_YAW_GAIN, -TURN_YAW_MAX_DEG, TURN_YAW_MAX_DEG))
        a = TURN_YAW_ALPHA
        self._turn_yaw_deg = (1.0 - a) * self._turn_yaw_deg + a * target
        if abs(self._turn_yaw_deg) < 0.12 and abs(target) < 0.12:
            self._turn_yaw_deg = 0.0
        ego_q = -self._turn_yaw_deg
        world_target = -0.35 * ego_q
        self._world_flash_deg = (1.0 - a) * self._world_flash_deg + a * world_target
        if abs(self._world_flash_deg) < 0.08 and abs(world_target) < 0.08:
            self._world_flash_deg = 0.0
        self._set("egoX", round(float(offset), 3))
        self._set("egoYawDeg", round(ego_q, 2))
        self._set("worldYawDeg", round(self._world_flash_deg, 2))
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
        alerts=None,
    ):
        self.proposals = proposals if proposals is not None else []
        self.processed_objs = processed_objs if processed_objs is not None else []
        self.cipo_status = cipo_status
        self.left_3d = left_3d
        self.right_3d = right_3d

        self._ego_speed_mps = None if speed_mps is None else float(speed_mps)
        self.lane_frame.update(left_3d, right_3d, speed_mps=speed_mps, dt=dt)
        kappa = (2.0 * float(self.lane_frame.curvature)) if self.lane_frame.valid else None
        self._ego_pose.update(dt, speed_mps, kappa=kappa)
        if not self.lane_frame.valid:
            self._slots.reset()
        slots, offsets = self._active_slots(self.proposals)

        self._push_ego_pose()
        self._push_cipo(self.processed_objs, cipo_status)
        alerts = alerts or {}
        self._set("ldwSide", str(alerts.get("ldw_side") or alerts.get("ldw") or ""))
        self._set("fcwLevel", str(alerts.get("fcw") or "OFF"))

        rows = self._traffic_payload(self.processed_objs)
        self._traffic_seed = rows
        self._traffic_t0 = time.perf_counter()
        payload = json.dumps(rows, separators=(",", ":"))
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
