"""
Qt Quick 3D BEV viewport.

Embeds a View3D inside the existing QWidget cockpit via QQuickWidget.
Same public API as BEVWidget so main_window can swap backends.
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
from src.utils.drivable_area import find_ego_lanes, find_outer_lanes, parse_lane_components

# Local copies — do not import bev_widget (QPainter + sprite pipeline).
BEV_MAX_DIST_M = 60.0
BEV_MAX_LATERAL_M = 14.0
DEFAULT_VIEW_PITCH = 31.0
DEFAULT_VIEW_YAW = 0.0
DEFAULT_ZOOM = 1.05
DEFAULT_CALIB_PITCH = -7.0
DEFAULT_CALIB_H = 1.0

_QML_PATH = os.path.join(os.path.dirname(__file__), "qml", "BevScene.qml")
_YAW_AWAY = 180.0  # same heading as ego (rear toward chase cam)
_SEDAN_CYCLE = (KIND_SKODA, KIND_SHC)


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
        self._motion = {}
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

    @staticmethod
    def _interp_x_at_y(pts, y_target):
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1:
            return None
        ys = arr[:, 1]
        xs = arr[:, 0]
        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]
        if y_target <= ys[0]:
            return float(xs[0])
        if y_target >= ys[-1]:
            return float(xs[-1])
        return float(np.interp(y_target, ys, xs))

    def _corridor_polylines(self, left_3d, right_3d, y_start=2.0):
        if left_3d is None or right_3d is None:
            return None, None
        left = np.asarray(left_3d, dtype=np.float64)
        right = np.asarray(right_3d, dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or len(left) < 2 or len(right) < 2:
            return None, None
        x_l0 = self._interp_x_at_y(left, y_start)
        x_r0 = self._interp_x_at_y(right, y_start)
        if x_l0 is None or x_r0 is None:
            return None, None
        if x_l0 > x_r0:
            x_l0, x_r0 = x_r0, x_l0
        left_out = [(x_l0, float(y_start))]
        right_out = [(x_r0, float(y_start))]
        for row in left:
            if float(row[1]) > y_start + 0.05:
                left_out.append((float(row[0]), float(row[1])))
        for row in right:
            if float(row[1]) > y_start + 0.05:
                right_out.append((float(row[0]), float(row[1])))
        if len(left_out) < 2 or len(right_out) < 2:
            return None, None
        return left_out, right_out

    @staticmethod
    def _pair_segments(left_xy, right_xy, max_segs=14):
        n = min(len(left_xy), len(right_xy), max_segs + 1)
        if n < 2:
            return []
        step = max(1, (n - 1) // max_segs)
        segs = []
        i = 0
        while i + step < n:
            (xl0, yl0), (xl1, yl1) = left_xy[i], left_xy[i + step]
            (xr0, yr0), (xr1, yr1) = right_xy[i], right_xy[i + step]
            x0 = 0.5 * (xl0 + xr0)
            x1 = 0.5 * (xl1 + xr1)
            y0 = 0.5 * (yl0 + yr0)
            y1 = 0.5 * (yl1 + yr1)
            dx = x1 - x0
            dy = y1 - y0
            length = math.hypot(dx, dy)
            if length < 0.4:
                i += step
                continue
            w0 = abs(xr0 - xl0)
            w1 = abs(xr1 - xl1)
            w = max(0.8, 0.5 * (w0 + w1))
            # Qt yaw: +Y up, length along local Z after yaw. World Δx, Δz=-Δy.
            yaw = math.degrees(math.atan2(dx, -dy))
            segs.append({
                "x": round(0.5 * (x0 + x1), 2),
                "z": round(-0.5 * (y0 + y1), 2),
                "yaw": round(yaw, 1),
                "len": round(length, 2),
                "w": round(w, 2),
            })
            i += step
        return segs

    @staticmethod
    def _visible_xy(xs, ys, vis):
        xy = []
        for i in range(len(xs)):
            if not vis[i]:
                continue
            y = float(ys[i])
            if y <= 1.0 or y > BEV_MAX_DIST_M:
                continue
            xy.append((float(xs[i]), y))
        return xy

    @staticmethod
    def _point_at_s(pts, cum, s):
        if s <= 0:
            return pts[0]
        if s >= cum[-1]:
            return pts[-1]
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = max(0, min(i, len(pts) - 2))
        span = cum[i + 1] - cum[i]
        t = 0.0 if span < 1e-6 else (s - cum[i]) / span
        return (1.0 - t) * pts[i] + t * pts[i + 1]

    def _dashes_along(self, xy, dash=3.2, gap=4.0, max_segs=12):
        if len(xy) < 2:
            return []
        pts = np.asarray(xy, dtype=np.float64)
        dxy = np.diff(pts, axis=0)
        seglen = np.hypot(dxy[:, 0], dxy[:, 1])
        total = float(seglen.sum())
        if total < 1.0:
            return []
        cum = np.concatenate(([0.0], np.cumsum(seglen)))
        period = dash + gap
        segs = []
        s0 = 0.0
        while s0 < total - 0.35 and len(segs) < max_segs:
            s1 = min(s0 + dash, total)
            if s1 - s0 >= 0.45:
                p0 = self._point_at_s(pts, cum, s0)
                p1 = self._point_at_s(pts, cum, s1)
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                length = math.hypot(dx, dy)
                if length >= 0.45:
                    yaw = math.degrees(math.atan2(dx, -dy))
                    segs.append({
                        "x": round(0.5 * (float(p0[0]) + float(p1[0])), 2),
                        "z": round(-0.5 * (float(p0[1]) + float(p1[1])), 2),
                        "yaw": round(yaw, 1),
                        "len": round(length, 2),
                        "w": 0.18,
                    })
            s0 += period
        return segs

    def _dash_payload(self, proposals):
        """White Film dashes along detected ego-left / ego-right polylines (not inset)."""
        rows = []
        if not proposals:
            return rows
        ego_left, ego_right = find_ego_lanes(proposals)
        for lane in (ego_left, ego_right):
            if lane is None:
                continue
            try:
                xs, ys, zs, vis = parse_lane_components(lane)
            except Exception:
                continue
            xy = self._visible_xy(xs, ys, vis)
            rows.extend(self._dashes_along(xy))
            if len(rows) >= 28:
                break
        return rows[:28]

    def _edge_payload(self, proposals):
        """
        P1: dynamic outer road-edge ribbons from outermost detected lanes.
        Falls back to empty → QML keeps static ±5.35 only when no detections.
        """
        rows = []
        if not proposals:
            return rows
        outer_l, outer_r = find_outer_lanes(proposals)
        for lane in (outer_l, outer_r):
            if lane is None:
                continue
            try:
                xs, ys, zs, vis = parse_lane_components(lane)
            except Exception:
                continue
            xy = self._visible_xy(xs, ys, vis)
            if len(xy) < 2:
                continue
            # Thicker white edge (w~0.22) along detected outer polyline
            n = len(xy)
            step = max(1, (n - 1) // 10)
            i = 0
            while i + step < n and len(rows) < 24:
                x0, y0 = xy[i]
                x1, y1 = xy[i + step]
                dx = x1 - x0
                dy = y1 - y0
                length = math.hypot(dx, dy)
                if length >= 0.6:
                    yaw = math.degrees(math.atan2(dx, -dy))
                    rows.append({
                        "x": round(0.5 * (x0 + x1), 2),
                        "z": round(-0.5 * (y0 + y1), 2),
                        "yaw": round(yaw, 1),
                        "len": round(length, 2),
                        "w": 0.22,
                    })
                i += step
        return rows[:24]

    @staticmethod
    def _ribbon_segments(xy, pal, max_segs=12):
        if len(xy) < 2:
            return []
        n = len(xy)
        step = max(1, (n - 1) // max_segs)
        segs = []
        i = 0
        while i + step < n:
            x0, y0 = xy[i]
            x1, y1 = xy[i + step]
            dx = x1 - x0
            dy = y1 - y0
            length = math.hypot(dx, dy)
            if length < 0.5:
                i += step
                continue
            yaw = math.degrees(math.atan2(dx, -dy))
            segs.append({
                "x": round(0.5 * (x0 + x1), 2),
                "z": round(-0.5 * (y0 + y1), 2),
                "yaw": round(yaw, 1),
                "len": round(length, 2),
                "w": 0.16,
                "c": pal,
            })
            i += step
        return segs

    def _corridor_payload(self, left_3d, right_3d):
        left_xy, right_xy = self._corridor_polylines(left_3d, right_3d)
        if not left_xy:
            return []
        return self._pair_segments(left_xy, right_xy)

    def _lane_payload(self, proposals):
        rows = []
        if not proposals:
            return rows
        n_lane = 0
        for lane in proposals:
            if n_lane >= 6:
                break
            try:
                xs, ys, zs, vis = parse_lane_components(lane)
            except Exception:
                continue
            if int(np.sum(vis)) < 2:
                continue
            xy = self._visible_xy(xs, ys, vis)
            if len(xy) < 2:
                continue
            mean_x = float(np.mean([p[0] for p in xy]))
            if abs(mean_x) > 7.5:
                continue
            if abs(mean_x) < 2.2:
                pal = 0
            elif mean_x < 0:
                pal = 1
            else:
                pal = 2
            segs = self._ribbon_segments(xy, pal)
            if not segs:
                continue
            rows.extend(segs)
            n_lane += 1
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
            x = float(obj["X_3d"])
            z = float(obj["Z_3d"])
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
        live = {int(o.get("track_id", -1)) for o in visible}
        self._motion = {k: v for k, v in self._motion.items() if k in live}
        return rows

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
        self._set("cipoX", round(float(cipo.get("X_3d", 0.0)), 2))
        self._set("cipoZ", round(-z, 2))
        self._set("cipoDist", round(z, 1))

    def update_bev_data(self, proposals, processed_objs=None, cipo_status="SAFE", left_3d=None, right_3d=None):
        self.proposals = proposals if proposals is not None else []
        self.processed_objs = processed_objs if processed_objs is not None else []
        self.cipo_status = cipo_status
        self.left_3d = left_3d
        self.right_3d = right_3d
        rows = self._traffic_payload(self.processed_objs)
        self._push_cipo(self.processed_objs, cipo_status)
        payload = json.dumps(rows, separators=(",", ":"))
        if payload != self._last_traffic_json:
            self._last_traffic_json = payload
            self._set("trafficJson", payload)
        corr = json.dumps(self._corridor_payload(left_3d, right_3d), separators=(",", ":"))
        if corr != self._last_corridor_json:
            self._last_corridor_json = corr
            self._set("corridorJson", corr)
        lanes = json.dumps(self._lane_payload(self.proposals), separators=(",", ":"))
        if lanes != self._last_lane_json:
            self._last_lane_json = lanes
            self._set("laneJson", lanes)
        dashes = json.dumps(self._dash_payload(self.proposals), separators=(",", ":"))
        if dashes != self._last_dash_json:
            self._last_dash_json = dashes
            self._set("dashJson", dashes)
        edges = json.dumps(self._edge_payload(self.proposals), separators=(",", ":"))
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
