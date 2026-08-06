"""
PySide6 Interactive 3D Rotatable Telemetry BEV Widget
Renders dynamic 3D perspective Bird's Eye View (BEV) map with:
 - 360° Mouse Yaw & Pitch Camera Rotation
 - Sleek Vector Sports Car Silhouette with Glowing Headlight Beams
 - 3D Perspective Distance Arcs & Gridlines
 - Translucent 3D Drivable Corridor Mesh & Glowing Lane Ribbons
 - 3D Positioned Vehicle Target Markers & ID Badges
"""

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QFont, QPainterPath,
    QLinearGradient, QRadialGradient
)

from src.inference.postprocess import ANCHOR_Y_STEPS
from src.utils.drivable_area import parse_lane_components

class BEVWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(450, 600)
        self.setMouseTracking(True)

        # ── 3D Interactive Camera State ──────────────────────────────────────
        self.pitch_deg = 35.0          # Pitch tilt (0 = top-down 2D, 60 = 3D horizon tilt)
        self.yaw_deg = 0.0             # Yaw rotation (-90 to +90 degrees)
        self.cam_h = 10.0              # Camera height above ground (meters)
        self.cam_dist = 12.0           # Camera distance behind ego vehicle (meters)
        self.zoom_factor = 1.0         # Zoom multiplier
        self.pan_offset = QPointF(0, 0)# Canvas translation

        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = QPointF(0, 0)

        # ── Current Render Data ──────────────────────────────────────────────
        self.proposals = []
        self.processed_objs = []
        self.cipo_status = "SAFE"
        self.left_3d = None
        self.right_3d = None

        # ── Calibration Offsets ──────────────────────────────────────────────
        self.calib_pitch = 0.0
        self.calib_h = 1.6

        # ── Theme Palette ────────────────────────────────────────────────────
        self.bg_color = QColor(14, 18, 24)
        self.road_color = QColor(24, 29, 38)
        self.grid_color = QColor(42, 50, 62)
        self.grid_text_color = QColor(130, 145, 165)

    def set_calibration(self, pitch_deg, height_m):
        """Updates extrinsics calibration offset."""
        self.calib_pitch = pitch_deg
        self.calib_h = height_m
        self.update()

    def update_bev_data(self, proposals, processed_objs=None, cipo_status="SAFE", left_3d=None, right_3d=None):
        """Receives new frame BEV data and triggers UI render update."""
        self.proposals = proposals if proposals is not None else []
        self.processed_objs = processed_objs if processed_objs is not None else []
        self.cipo_status = cipo_status
        self.left_3d = left_3d
        self.right_3d = right_3d
        self.update()

    # ─────────────────────────────────────────────────────────────────────────
    # 3D PERSPECTIVE PROJECTION MATH ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def world_to_canvas_3d(self, x, y, z=0.0, width=None, height=None):
        """
        Projects 3D world coordinate (X=lateral m, Y=forward m, Z=height m)
        onto 2D viewport pixels using a 3D perspective camera matrix with Pitch & Yaw.
        """
        w = width if width is not None else self.width()
        h = height if height is not None else self.height()

        focal = 460.0 * self.zoom_factor

        # Relative coordinate to camera origin
        dx = x
        dy = y + self.cam_dist
        # Apply camera height calibration scaling (scaled relative to base 1.6m height)
        dz = z - (self.cam_h * (self.calib_h / 1.6))

        # 1. Yaw rotation (around Y axis)
        rad_y = np.radians(self.yaw_deg)
        rx = dx * np.cos(rad_y) - dy * np.sin(rad_y)
        ry = dx * np.sin(rad_y) + dy * np.cos(rad_y)
        rz = dz

        # 2. Pitch rotation (tilt down around X axis, incorporating pitch calibration)
        rad_p = np.radians(self.pitch_deg + self.calib_pitch)
        cy = ry * np.cos(rad_p) - rz * np.sin(rad_p)
        cz = ry * np.sin(rad_p) + rz * np.cos(rad_p)
        cx = rx

        if cy <= 0.2:
            return None

        # 3. Perspective Projection & Viewport Scaling
        px = (w / 2.0) + (cx / cy) * focal + self.pan_offset.x()
        py = (h * 0.72) - (cz / cy) * focal + self.pan_offset.y()

        return QPointF(px, py)

    # ─────────────────────────────────────────────────────────────────────────
    # MOUSE INTERACTION HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_rotating = True
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        pos = event.position()
        delta = pos - self.last_mouse_pos
        self.last_mouse_pos = pos

        if self.is_rotating:
            # Horizontal drag = Yaw rotation, Vertical drag = Pitch tilt
            self.yaw_deg = max(-85.0, min(85.0, self.yaw_deg + delta.x() * 0.35))
            self.pitch_deg = max(5.0, min(75.0, self.pitch_deg - delta.y() * 0.35))
            self.update()
        elif self.is_panning:
            self.pan_offset += delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_rotating = False
        elif event.button() == Qt.RightButton:
            self.is_panning = False

    def wheelEvent(self, event):
        angle_delta = event.angleDelta().y()
        factor = 1.12 if angle_delta > 0 else 0.88
        new_zoom = self.zoom_factor * factor
        if 0.4 <= new_zoom <= 4.5:
            self.zoom_factor = new_zoom
            self.update()

    def reset_view(self):
        """Resets camera to standard 3D perspective."""
        self.pitch_deg = 35.0
        self.yaw_deg = 0.0
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.update()

    # ─────────────────────────────────────────────────────────────────────────
    # CANVAS RENDERER
    # ─────────────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter()
        if not painter.begin(self):
            return

        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

            w = self.width()
            h = self.height()

            # ── 1. Background Fill ───────────────────────────────────────────
            painter.fillRect(self.rect(), self.bg_color)

            # ── 2. Asphalt Road Corridor Mesh ────────────────────────────────
            road_pts_3d = [(-6.0, 0.0), (+6.0, 0.0), (+6.0, 80.0), (-6.0, 80.0)]
            road_pts_2d = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y in road_pts_3d]
            if all(p is not None for p in road_pts_2d):
                road_poly = QPolygonF(road_pts_2d)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(self.road_color))
                painter.drawPolygon(road_poly)

            # ── 3. 3D Perspective Distance Arcs & Gridlines ─────────────────
            painter.setFont(QFont("Inter", 8))
            grid_pen = QPen(self.grid_color, 1, Qt.DotLine)
            painter.setPen(grid_pen)

            # Distance Arcs (10m, 20m, 30m, 40m, 50m, 60m, 70m, 80m)
            for y_m in range(10, 85, 10):
                arc_pts = []
                for x_m in np.linspace(-7.0, 7.0, 15):
                    pt = self.world_to_canvas_3d(x_m, float(y_m), 0.0, w, h)
                    if pt is not None:
                        arc_pts.append(pt)
                if len(arc_pts) > 1:
                    path = QPainterPath()
                    path.moveTo(arc_pts[0])
                    for p in arc_pts[1:]:
                        path.lineTo(p)
                    painter.setPen(grid_pen)
                    painter.drawPath(path)

                    # Distance Label
                    mid_pt = arc_pts[len(arc_pts) // 2]
                    painter.setPen(QPen(self.grid_text_color))
                    painter.drawText(int(mid_pt.x() - 14), int(mid_pt.y() - 3), f"{y_m}m")

            # Lateral Gridlines (-4m, -2m, 0m, +2m, +4m)
            for x_m in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                line_pts = []
                for y_m in np.linspace(0.0, 80.0, 10):
                    pt = self.world_to_canvas_3d(x_m, y_m, 0.0, w, h)
                    if pt is not None:
                        line_pts.append(pt)
                if len(line_pts) > 1:
                    path = QPainterPath()
                    path.moveTo(line_pts[0])
                    for p in line_pts[1:]:
                        path.lineTo(p)
                    line_pen = QPen(QColor(0, 200, 255, 160), 1.5, Qt.SolidLine) if x_m == 0.0 else grid_pen
                    painter.setPen(line_pen)
                    painter.drawPath(path)

            # ── 4. 3D Drivable Area Corridor Mesh ─────────────────────────────
            if self.left_3d is not None and self.right_3d is not None and len(self.left_3d) > 0 and len(self.right_3d) > 0:
                pts_left  = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y, _ in self.left_3d]
                pts_right = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y, _ in self.right_3d]
                pts_left  = [p for p in pts_left if p is not None]
                pts_right = [p for p in pts_right if p is not None]

                if len(pts_left) > 1 and len(pts_right) > 1:
                    corridor_poly = QPolygonF(pts_left + pts_right[::-1])
                    corridor_color = QColor(220, 30, 30, 85) if self.cipo_status == "DANGER" else QColor(0, 220, 100, 75)
                    painter.setBrush(QBrush(corridor_color))
                    painter.setPen(QPen(corridor_color.lighter(130), 1.0, Qt.SolidLine))
                    painter.drawPolygon(corridor_poly)

            # ── 5. 3D Glowing Lane Line Polylines ─────────────────────────
            if self.proposals:
                for lane in self.proposals:
                    xs, ys, zs, vis = parse_lane_components(lane)
                    if vis.sum() < 2:
                        continue

                    pts_world = [(xs[i], ys[i]) for i in range(len(xs)) if vis[i]]
                    mean_x = float(np.mean([wx for wx, wy in pts_world]))

                    # ── Skip far outer lanes (beyond ±1 adjacent lane) ──────
                    if abs(mean_x) > 7.5:
                        continue

                    pts_canvas = [self.world_to_canvas_3d(wx, wy, 0.0, w, h) for wx, wy in pts_world]
                    valid_pts  = [p for p in pts_canvas if p is not None]

                    if len(valid_pts) > 1:
                        if abs(mean_x) < 2.0:
                            lane_color = QColor(0, 220, 255)   # Ego lane Cyan
                        elif mean_x < 0:
                            lane_color = QColor(255, 190, 0)   # Left adjacent Gold
                        else:
                            lane_color = QColor(0, 255, 180)   # Right adjacent Light Green

                        lane_pen = QPen(lane_color, 2.5, Qt.SolidLine)
                        painter.setPen(lane_pen)

                        path = QPainterPath()
                        path.moveTo(valid_pts[0])
                        for pt in valid_pts[1:]:
                            path.lineTo(pt)
                        painter.drawPath(path)

                        # Node circles along lane polylines
                        painter.setBrush(QBrush(QColor(255, 255, 255)))
                        painter.setPen(Qt.NoPen)
                        for pt in valid_pts[::3]:
                            painter.drawEllipse(pt, 2.0, 2.0)

            # ── 6. Headlight Light Cones (Ego Vehicle Headlamp Beams) ─────────
            beam_left  = self.world_to_canvas_3d(-0.9, 0.5, 0.0, w, h)
            beam_right = self.world_to_canvas_3d(+0.9, 0.5, 0.0, w, h)
            far_left   = self.world_to_canvas_3d(-3.5, 28.0, 0.0, w, h)
            far_right  = self.world_to_canvas_3d(+3.5, 28.0, 0.0, w, h)

            if all(p is not None for p in [beam_left, beam_right, far_left, far_right]):
                beam_poly = QPolygonF([beam_left, far_left, far_right, beam_right])
                beam_grad = QLinearGradient(beam_left, far_left)
                beam_grad.setColorAt(0.0, QColor(0, 240, 255, 45))
                beam_grad.setColorAt(1.0, QColor(0, 240, 255, 0))
                painter.setBrush(QBrush(beam_grad))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(beam_poly)

            # ── 7. Ego Vehicle — Premium Sports Sedan Top-Down Vector ──────────
            # Body outline (sporty aerodynamics with side mirrors, wide wheel arches)
            car_body_pts = [
                # Front center bumper nose
                (0.00,  2.45),
                # Front hood curves (front-right)
                (0.35,  2.41), (0.62,  2.28), (0.75,  2.05),
                # Wide Front Wheel Arch
                (0.88,  1.90), (0.94,  1.60), (0.90,  1.20),
                # Right Side Mirror
                (1.12,  1.05), (1.15,  0.88), (0.92,  0.86),
                # Right Door / Midsection curve
                (0.92,  0.20), (0.92, -0.60),
                # Wide Rear Wheel Arch
                (0.94, -0.80), (0.98, -1.30), (0.92, -1.65),
                # Rear bumper & diffuser (right)
                (0.80, -2.15), (0.45, -2.35),
                # Rear diffuser center (narrow exhaust/vent channel)
                (0.00, -2.40),
                # Rear bumper & diffuser (left)
                (-0.45, -2.35), (-0.80, -2.15),
                # Wide Rear Wheel Arch (left)
                (-0.92, -1.65), (-0.98, -1.30), (-0.94, -0.80),
                # Left Door / Midsection curve
                (-0.92, -0.60), (-0.92,  0.20),
                # Left Side Mirror
                (-0.92,  0.86), (-1.15,  0.88), (-1.12,  1.05),
                # Wide Front Wheel Arch (left)
                (-0.90,  1.20), (-0.94,  1.60), (-0.88,  1.90),
                # Front hood curves (front-left)
                (-0.75,  2.05), (-0.62,  2.28), (-0.35,  2.41),
            ]
            body_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in car_body_pts]
            if all(p is not None for p in body_canvas):
                body_path = QPainterPath()
                body_path.moveTo(body_canvas[0])
                for bp in body_canvas[1:]:
                    body_path.lineTo(bp)
                body_path.closeSubpath()

                # Premium metallic titanium-grey body fill with glowing cyan outline
                painter.setPen(QPen(QColor(0, 220, 255, 220), 1.8, Qt.SolidLine))
                body_grad = QLinearGradient(body_canvas[0], body_canvas[len(body_canvas)//2])
                body_grad.setColorAt(0.0, QColor(25, 35, 48))
                body_grad.setColorAt(0.5, QColor(16, 22, 32))
                body_grad.setColorAt(1.0, QColor(10, 14, 20))
                painter.setBrush(QBrush(body_grad))
                painter.drawPath(body_path)

            # ----- Dual Racing / Hood Accent Stripes -----
            for offset_x in [-0.22, 0.12]:
                stripe_pts = [
                    (offset_x, 2.30), (offset_x + 0.10, 2.30),
                    (offset_x + 0.10, 0.90), (offset_x, 0.90)
                ]
                stripe_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in stripe_pts]
                if all(p is not None for p in stripe_canvas):
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(0, 220, 255, 60)))  # Subtle glowing racing stripes
                    painter.drawPolygon(QPolygonF(stripe_canvas))

            # ----- Premium Cabin Greenhouse Glass (windscreens & side panels) -----
            roof_pts = [
                (-0.68,  0.85), (+0.68,  0.85),
                (+0.75,  0.20), (+0.75, -0.65),
                (-0.75, -0.65), (-0.75,  0.20),
            ]
            roof_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in roof_pts]
            if all(p is not None for p in roof_canvas):
                roof_poly = QPolygonF(roof_canvas)
                painter.setPen(QPen(QColor(0, 160, 255, 180), 1.0))
                painter.setBrush(QBrush(QColor(24, 32, 45)))
                painter.drawPolygon(roof_poly)

            # ----- Aerodynamic Windshield (Reflective Glass) -----
            ws_pts = [(-0.60, 0.83), (+0.60, 0.83), (+0.70, 0.25), (-0.70, 0.25)]
            ws_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in ws_pts]
            if all(p is not None for p in ws_canvas):
                ws_grad = QLinearGradient(ws_canvas[0], ws_canvas[2])
                ws_grad.setColorAt(0.0, QColor(0, 180, 255, 120))
                ws_grad.setColorAt(1.0, QColor(0, 100, 180, 80))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(ws_grad))
                painter.drawPolygon(QPolygonF(ws_canvas))

            # ----- Rear glass window -----
            rw_pts = [(-0.58, -0.67), (+0.58, -0.67), (+0.64, -1.35), (-0.64, -1.35)]
            rw_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in rw_pts]
            if all(p is not None for p in rw_canvas):
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(30, 80, 140, 100)))
                painter.drawPolygon(QPolygonF(rw_canvas))

            # ----- Wheels with performance brake calipers -----
            wheel_defs = [
                [(-0.96, 1.60), (-0.96, 1.10), (-0.80, 1.10), (-0.80, 1.60)],  # FL
                [(+0.80, 1.60), (+0.80, 1.10), (+0.96, 1.10), (+0.96, 1.60)],  # FR
                [(-0.96,-1.00), (-0.96,-1.50), (-0.80,-1.50), (-0.80,-1.00)],  # RL
                [(+0.80,-1.00), (+0.80,-1.50), (+0.96,-1.50), (+0.96,-1.00)],  # RR
            ]
            for wdef in wheel_defs:
                wc = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in wdef]
                if all(p is not None for p in wc):
                    painter.setPen(QPen(QColor(40, 48, 56), 1.0))
                    painter.setBrush(QBrush(QColor(22, 26, 30)))
                    painter.drawPolygon(QPolygonF(wc))

                    # Inner silver alloy spokes/ring details
                    cx_w = float(np.mean([px for px, py in wdef]))
                    cy_w = float(np.mean([py for px, py in wdef]))
                    center = self.world_to_canvas_3d(cx_w, cy_w, 0.0, w, h)
                    if center is not None:
                        painter.setPen(QPen(QColor(150, 160, 175), 1.2))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(center, 3.8, 3.8)

            # ----- Triple-beam LED Projection Headlamps -----
            for hx, hy in [(-0.58, 2.38), (+0.58, 2.38)]:
                pt = self.world_to_canvas_3d(hx, hy, 0.0, w, h)
                if pt is not None:
                    # Glow halo
                    glow = QRadialGradient(pt, 8)
                    glow.setColorAt(0.0, QColor(255, 255, 220, 200))
                    glow.setColorAt(1.0, QColor(255, 255, 200, 0))
                    painter.setBrush(QBrush(glow))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(pt, 8.0, 8.0)
                    # Bright core
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.drawEllipse(pt, 3.0, 3.0)

            # ----- LED Tail lights (rear) ------------------------------------
            for tx, ty in [(-0.72, -2.32), (+0.72, -2.32)]:
                pt = self.world_to_canvas_3d(tx, ty, 0.0, w, h)
                if pt is not None:
                    glow = QRadialGradient(pt, 7)
                    glow.setColorAt(0.0, QColor(255, 40, 40, 220))
                    glow.setColorAt(1.0, QColor(255, 0, 0, 0))
                    painter.setBrush(QBrush(glow))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(pt, 7.0, 7.0)
                    painter.setBrush(QBrush(QColor(255, 30, 30)))
                    painter.drawEllipse(pt, 2.8, 2.8)

            # ── 8. Render 3D Target Vehicle Mini-Car Icons ──────────────────
            if self.processed_objs:
                for obj in self.processed_objs:
                    x_3d = obj['X_3d']
                    z_3d = obj['Z_3d']
                    # Only render vehicles in valid forward range and within ±2 lanes
                    if not (0 < z_3d <= 80.0):
                        continue
                    if abs(x_3d) > 9.0:          # skip objects beyond ±2 lanes
                        continue

                    pt_obj = self.world_to_canvas_3d(x_3d, z_3d, 0.0, w, h)
                    if pt_obj is None:
                        continue
                    if not (0 <= pt_obj.x() <= w and 0 <= pt_obj.y() <= h):
                        continue

                    is_cipo = obj.get('is_cipo', False)
                    in_path = obj.get('in_path', False)
                    dist_z  = obj['Z_3d']

                    if is_cipo or dist_z < 15.0:
                        marker_color = QColor(255, 40, 40)    # RED  — Danger / CIPO
                    elif in_path:
                        marker_color = QColor(255, 200, 0)    # GOLD — In-path warning
                    else:
                        marker_color = QColor(0, 220, 255)    # CYAN — Adjacent safe

                    # ── Mini top-down car silhouette for target vehicles ─────
                    # Half-widths in world meters (smaller than ego car)
                    hw, hl = 0.70, 1.40
                    mini_body = [
                        (x_3d - 0.55, z_3d + hl),   # Front-left
                        (x_3d + 0.55, z_3d + hl),   # Front-right
                        (x_3d + hw,   z_3d + hl*0.5),
                        (x_3d + hw,   z_3d - hl*0.5),
                        (x_3d + 0.50, z_3d - hl),   # Rear-right
                        (x_3d - 0.50, z_3d - hl),   # Rear-left
                        (x_3d - hw,   z_3d - hl*0.5),
                        (x_3d - hw,   z_3d + hl*0.5),
                    ]
                    mini_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in mini_body]
                    mini_canvas = [p for p in mini_canvas if p is not None]
                    if len(mini_canvas) >= 4:
                        # Body fill
                        body_color = QColor(marker_color.red(), marker_color.green(),
                                            marker_color.blue(), 160)
                        painter.setBrush(QBrush(body_color))
                        painter.setPen(QPen(marker_color, 1.5, Qt.SolidLine))
                        mini_path = QPainterPath()
                        mini_path.moveTo(mini_canvas[0])
                        for mp in mini_canvas[1:]:
                            mini_path.lineTo(mp)
                        mini_path.closeSubpath()
                        painter.drawPath(mini_path)

                        # Windshield highlight
                        ws_l = self.world_to_canvas_3d(x_3d - 0.40, z_3d + hl * 0.55, 0.0, w, h)
                        ws_r = self.world_to_canvas_3d(x_3d + 0.40, z_3d + hl * 0.55, 0.0, w, h)
                        ws_bl = self.world_to_canvas_3d(x_3d - 0.40, z_3d + hl * 0.20, 0.0, w, h)
                        ws_br = self.world_to_canvas_3d(x_3d + 0.40, z_3d + hl * 0.20, 0.0, w, h)
                        if all(p is not None for p in [ws_l, ws_r, ws_bl, ws_br]):
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QBrush(QColor(180, 230, 255, 130)))
                            painter.drawPolygon(QPolygonF([ws_l, ws_r, ws_br, ws_bl]))

                        # Front headlight dots
                        for hx, hy in [(x_3d - 0.48, z_3d + hl * 0.92),
                                       (x_3d + 0.48, z_3d + hl * 0.92)]:
                            ht = self.world_to_canvas_3d(hx, hy, 0.0, w, h)
                            if ht is not None:
                                painter.setBrush(QBrush(QColor(255, 255, 200)))
                                painter.setPen(Qt.NoPen)
                                painter.drawEllipse(ht, 2.5, 2.5)

                    # ── Distance & ID label badge ────────────────────────────
                    track_id = obj.get('track_id', -1)
                    id_str   = f"#{track_id:02d} " if track_id > 0 else ""
                    label_str = f"{id_str}{obj['label'].upper()} {dist_z:.1f}m"

                    # Badge background pill
                    badge_w = len(label_str) * 5 + 10
                    badge_x = int(pt_obj.x() + hl * 8 + 4)
                    badge_y = int(pt_obj.y() - 8)
                    painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(badge_x, badge_y, badge_w, 16, 4, 4)

                    painter.setFont(QFont("Inter", 7, QFont.Bold))
                    painter.setPen(QPen(marker_color.lighter(130)))
                    painter.drawText(badge_x + 4, badge_y + 11, label_str)

            # ── 9. Top-Left Telemetry Overlay Badge ──────────────────────────
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Inter", 10, QFont.Bold))
            painter.drawText(15, 28, "3D ROTATABLE BEV TELEMETRY CANVAS")

            badge_color = QColor(220, 40, 40) if self.cipo_status == "DANGER" else (QColor(255, 180, 0) if self.cipo_status == "WARNING" else QColor(0, 200, 100))
            painter.setBrush(QBrush(badge_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(15, 38, 110, 22, 4, 4)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Inter", 8, QFont.Bold))
            painter.drawText(23, 53, f"CIPO: {self.cipo_status}")

            # ── 10. Bottom-Right Controls Telemetry Legend ───────────────────
            painter.setFont(QFont("Inter", 8))
            painter.setPen(QPen(self.grid_text_color))
            painter.drawText(w - 240, h - 15, f"Pitch: {self.pitch_deg:.0f}° | Yaw: {self.yaw_deg:.0f}° | Drag to Rotate 3D")

        finally:
            painter.end()
