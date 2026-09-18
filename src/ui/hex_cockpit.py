"""Hexagonal infotainment shell: honeycomb bezel + cluster HUD over BEV / live."""

from __future__ import annotations

import math
from datetime import datetime

from PySide6.QtCore import (
    QEasingCurve,
    Property,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygon,
    QRegion,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

NAVY = QColor("#050A14")
CYAN = QColor("#3EC8FF")
ICE = QColor("#E8F4FF")
MUTED = QColor("#7A93A8")
WINDOW_W, WINDOW_H = 1280, 720


def _hex_metrics(w, h):
    """Inset hex so top/side bezels can hold speed, limit, and gear."""
    mx, my = w * 0.06, h * 0.12
    cx, cy = w * 0.5, h * 0.50
    return cx, cy, (w * 0.5) - mx, (h * 0.5) - my


def _flat_hex_points(cx, cy, rx, ry):
    pts = []
    for i in range(6):
        ang = math.radians(i * 60)
        pts.append(QPoint(int(cx + rx * math.cos(ang)), int(cy + ry * math.sin(ang))))
    return pts


def _set_text(label, text):
    if label.text() != text:
        label.setText(text)


class HexBezel(QWidget):
    """Navy honeycomb surround; pixmap-cached so video frames do not redraw it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._hex_pts = []
        self._cache = None

    def hex_polygon(self):
        return QPolygon(self._hex_pts) if self._hex_pts else QPolygon()

    def resizeEvent(self, event):
        self._rebuild()
        super().resizeEvent(event)

    def _rebuild(self):
        w, h = self.width(), self.height()
        if w < 8 or h < 8:
            return
        cx, cy, rx, ry = _hex_metrics(w, h)
        self._hex_pts = _flat_hex_points(cx, cy, rx, ry)
        inset = _flat_hex_points(cx, cy, rx * 0.988, ry * 0.988)
        self.setMask(QRegion(self.rect()) - QRegion(QPolygon(inset)))
        self._cache = QPixmap(w, h)
        self._cache.fill(NAVY)
        p = QPainter(self._cache)
        p.setRenderHint(QPainter.Antialiasing, True)
        self._paint_honeycomb(p, w, h)
        hex_path = QPainterPath()
        hex_path.addPolygon(QPolygon(self._hex_pts))
        p.end()
        self.update()

    def paintEvent(self, event):
        if self._cache is None:
            return
        QPainter(self).drawPixmap(0, 0, self._cache)

    def _paint_honeycomb(self, painter, w, h):
        r = 18.0
        dx = r * math.sqrt(3)
        dy = r * 1.5
        painter.setPen(QPen(QColor(40, 90, 130, 55), 1))
        painter.setBrush(Qt.NoBrush)
        rows = int(h / dy) + 2
        cols = int(w / dx) + 2
        for row in range(rows):
            oy = row * dy
            ox0 = dx * 0.5 if row % 2 else 0.0
            for col in range(cols):
                x = col * dx + ox0
                y = oy
                path = QPainterPath()
                for i in range(6):
                    ang = math.radians(30 + i * 60)
                    px = x + r * math.cos(ang)
                    py = y + r * math.sin(ang)
                    if i == 0:
                        path.moveTo(px, py)
                    else:
                        path.lineTo(px, py)
                path.closeSubpath()
                painter.drawPath(path)


class HexStroke(QWidget):
    """Unmasked 6-edge outline so top-left / top-right facets stay visible over video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._pts = []

    def set_points(self, pts):
        self._pts = list(pts)
        self.update()

    def paintEvent(self, event):
        if len(self._pts) < 6:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addPolygon(QPolygon(self._pts))
        path.closeSubpath()
        glow = QColor(CYAN)
        for width, alpha in ((7, 40), (3.2, 230)):
            glow.setAlpha(alpha)
            p.setPen(QPen(glow, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        p.end()


class SpeedGauge(QWidget):
    """Open horseshoe cluster gauge with tick marks (concept-art style)."""

    _START = 210.0
    _SPAN = -240.0
    _VMAX = 160.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kmh = 0.0
        self._shown = 0
        self.setFixedSize(200, 168)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_kmh(self, kmh):
        v = 0.0 if kmh is None else max(0.0, float(kmh))
        shown = int(round(v))
        if shown == self._shown:
            return
        self._shown = shown
        self._kmh = v
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2.0, 92.0
        r = 78.0
        ring = QRectF(cx - r, cy - r, r * 2, r * 2)
        start = int(self._START * 16)
        span = int(self._SPAN * 16)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(80, 170, 220, 70), 2))
        p.drawArc(ring, start, span)

        inner = ring.adjusted(9, 9, -9, -9)
        p.setPen(QPen(QColor(190, 225, 250, 210), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(inner, start, span)

        for i in range(25):
            t = i / 24.0
            deg = self._START + self._SPAN * t
            major = (i % 4 == 0)
            rad = math.radians(deg)
            r0 = r - (17 if major else 12)
            r1 = r - 5
            x0 = cx + r0 * math.cos(rad)
            y0 = cy - r0 * math.sin(rad)
            x1 = cx + r1 * math.cos(rad)
            y1 = cy - r1 * math.sin(rad)
            p.setPen(QPen(QColor(200, 230, 255, 230 if major else 140), 2 if major else 1))
            p.drawLine(QPoint(int(x0), int(y0)), QPoint(int(x1), int(y1)))

        frac = min(1.0, self._kmh / self._VMAX)
        if frac > 0.01:
            fill = ring.adjusted(5, 5, -5, -5)
            fill_span = int(self._SPAN * frac * 16)
            p.setPen(QPen(QColor(40, 160, 255, 90), 14, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(fill, start, fill_span)
            p.setPen(QPen(CYAN, 7, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(fill, start, fill_span)

        p.setPen(ICE)
        p.setFont(QFont("Segoe UI", 34, QFont.Bold))
        p.drawText(QRect(0, 52, self.width(), 50), Qt.AlignCenter, f"{self._shown}")
        p.setPen(QColor("#9EC8E0"))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRect(0, 100, self.width(), 18), Qt.AlignCenter, "km/h")
        p.end()


class TriggerBanner(QWidget):
    """Edge HUD chip: FCW (left) or LDW (right). Hidden until the tracker fires."""

    def __init__(self, kind="fcw", parent=None):
        super().__init__(parent)
        self._kind = kind
        self._title = "FCW" if kind == "fcw" else "LDW"
        self._body = ""
        self._pulse = 0.0
        self.setFixedSize(168, 72)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def _tick(self):
        self._pulse = (self._pulse + 0.12) % (2.0 * math.pi)
        self.update()

    def set_trigger(self, on, title=None, body=""):
        if title:
            self._title = title
        self._body = body or ""
        if on:
            if not self.isVisible():
                self.show()
                self._timer.start(50)
            self.raise_()
            self.update()
        elif self.isVisible():
            self.hide()
            self._timer.stop()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = 34, self.height() / 2.0
        hot = self._kind == "fcw"
        ring = QColor(255, 90, 40, int(50 + 40 * abs(math.sin(self._pulse)))) if hot else QColor(
            255, 180, 50, int(50 + 40 * abs(math.sin(self._pulse)))
        )
        for i, rad in enumerate((30, 24, 18)):
            c = QColor(ring)
            c.setAlpha(max(20, ring.alpha() - i * 18))
            p.setPen(QPen(c, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(cx - rad, cy - rad * 0.55, rad * 2.4, rad * 1.1), 10, 10)
        card = QRectF(8, 10, self.width() - 14, self.height() - 20)
        path = QPainterPath()
        path.addRoundedRect(card, 10, 10)
        border = QColor("#FF5A28") if hot else QColor("#FFC14A")
        p.setPen(QPen(border, 2))
        p.setBrush(QColor(28, 10, 8, 210) if hot else QColor(28, 20, 8, 210))
        p.drawPath(path)
        p.setPen(border)
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        p.drawText(QRect(44, 14, 120, 22), Qt.AlignLeft | Qt.AlignVCenter, self._title)
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor("#FFD0C0") if hot else QColor("#FFE6B0"))
        p.drawText(QRect(44, 36, 120, 20), Qt.AlignLeft | Qt.AlignVCenter, self._body)
        # shield / chevron mark
        p.setPen(QPen(border, 2))
        p.setBrush(Qt.NoBrush)
        if hot:
            p.drawRoundedRect(QRectF(16, 24, 18, 20), 3, 3)
        else:
            p.drawLine(QPoint(18, 40), QPoint(25, 28))
            p.drawLine(QPoint(25, 28), QPoint(32, 40))
        p.end()


class SettingsGear(QWidget):
    """Holographic rotating cog on the right hex edge."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(118, 118)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._angle = 0.0
        self._fast = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def set_active(self, on):
        self._fast = bool(on)
        self.update()

    def _tick(self):
        self._angle = (self._angle + (4.2 if self._fast else 1.1)) % 360.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(62, 200, 255, 50), 2))
        p.drawEllipse(QRectF(cx - 54, cy - 54, 108, 108))
        p.setPen(QPen(QColor(62, 200, 255, 90), 1.6))
        p.drawEllipse(QRectF(cx - 42, cy - 42, 84, 84))
        p.save()
        p.translate(cx, cy)
        p.rotate(-self._angle * 0.55)
        p.setPen(QPen(QColor(62, 200, 255, 80), 1.4))
        p.drawArc(QRectF(-50, -50, 100, 100), 40 * 16, 110 * 16)
        p.drawArc(QRectF(-50, -50, 100, 100), 220 * 16, 70 * 16)
        p.restore()
        p.save()
        p.translate(cx, cy)
        p.rotate(self._angle)
        teeth = 8
        r_out, r_in, r_hub = 28.0, 20.0, 8.0
        path = QPainterPath()
        for i in range(teeth * 2):
            ang = math.radians(i * 180.0 / teeth)
            r = r_out if i % 2 == 0 else r_in
            x, y = r * math.cos(ang), r * math.sin(ang)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        p.setPen(QPen(CYAN, 2.2))
        p.setBrush(QColor(20, 80, 130, 80))
        p.drawPath(path)
        p.setBrush(QColor(8, 20, 36, 180))
        p.drawEllipse(QPoint(0, 0), int(r_hub), int(r_hub))
        p.setPen(QPen(CYAN, 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPoint(0, 0), int(r_hub - 2), int(r_hub - 2))
        p.restore()
        p.end()


class SettingsWheel(QWidget):
    """Frosted cards that rise from the bottom; scroll to choose, tap to confirm."""

    confirmed = Signal(str)

    ITEMS = (
        ("night", "NIGHT MODE"),
        ("grid", "GRID HUD"),
        ("bev", "ADAS BEV"),
        ("live", "LIVE VIEW"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._index = 2
        self._rise = 0.0
        self._hex = []
        self._anim = None
        self.hide()

    def _get_rise(self):
        return self._rise

    def _set_rise(self, v):
        self._rise = float(v)
        self.update()

    rise = Property(float, _get_rise, _set_rise)

    def set_hex(self, pts):
        self._hex = list(pts)

    def open_menu(self):
        self.show()
        self.raise_()
        if self._anim is not None:
            self._anim.stop()
        self._rise = 0.0
        self._anim = QPropertyAnimation(self, b"rise", self)
        self._anim.setDuration(480)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def close_menu(self):
        if self._anim is not None:
            self._anim.stop()
        self.hide()
        self._rise = 0.0

    def wheelEvent(self, event):
        dy = event.angleDelta().y()
        if dy > 0:
            self._index = max(0, self._index - 1)
        elif dy < 0:
            self._index = min(len(self.ITEMS) - 1, self._index + 1)
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        key, _ = self.ITEMS[self._index]
        self.confirmed.emit(key)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if len(self._hex) >= 6:
            clip = QPainterPath()
            clip.addPolygon(QPolygon(self._hex))
            p.setClipPath(clip)
        cx = self.width() / 2.0
        cy = self.height() * 0.50
        n = len(self.ITEMS)
        order = [i for i in range(n) if i != self._index] + [self._index]
        for i in order:
            self._paint_card(p, cx, cy, i, i - self._index)
        p.setClipping(False)
        p.setPen(QColor(180, 220, 245, int(210 * self._rise)))
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        p.drawText(
            QRect(0, int(self.height() * 0.78), self.width(), 22),
            Qt.AlignCenter,
            "SCROLL TO CHOOSE  ·  TAP TO CONFIRM",
        )
        p.end()

    def _paint_card(self, p, cx, cy, i, rel):
        t = self._rise
        # Cards start below the hex and rise into a stacked wheel.
        y = cy + rel * 58.0 + (1.0 - t) * (220.0 + abs(rel) * 36.0)
        focus = rel == 0
        scale = 0.82 + 0.18 * t
        if rel < 0:
            w, h = (292.0 + (18 if rel == -1 else 0)) * scale, 44.0 * scale
        elif rel > 0:
            w, h = 400.0 * scale, 68.0 * scale
            y += 10
        else:
            w, h = 430.0 * scale, 86.0 * scale
        alpha_fill = int((85 if focus else 42) * t)
        alpha_border = int((240 if focus else 100) * t)
        rect = QRectF(cx - w / 2.0, y - h / 2.0, w, h)
        path = QPainterPath()
        path.addRoundedRect(rect, 10, 10)
        p.setPen(QPen(QColor(90, 210, 255, alpha_border), 2.4 if focus else 1.3))
        p.setBrush(QColor(10, 32, 52, alpha_fill))
        p.drawPath(path)
        if focus and t > 0.2:
            p.setPen(QPen(QColor(62, 200, 255, 55), 10))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.setPen(QPen(QColor(90, 210, 255, alpha_border), 2.4))
            p.setBrush(QColor(10, 32, 52, alpha_fill))
            p.drawPath(path)
        key, label = self.ITEMS[i]
        p.setPen(QColor(230, 245, 255, int(245 * t)))
        if focus:
            p.setFont(QFont("Segoe UI", 16, QFont.Bold))
            p.drawText(rect.adjusted(40, 4, -12, -28), Qt.AlignVCenter | Qt.AlignLeft, "✓  " + label)
            p.setFont(QFont("Segoe UI", 10))
            p.setPen(QColor(160, 210, 235, int(210 * t)))
            if key == "bev":
                p.drawText(rect.adjusted(68, 30, -12, -4), Qt.AlignLeft | Qt.AlignVCenter, "LANE TRACK")
        else:
            p.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
            p.drawText(rect, Qt.AlignCenter, label)


class HexCockpit(QWidget):
    view_changed = Signal(str)

    def __init__(self, bev_widget, parent=None):
        super().__init__(parent)
        self.setObjectName("hex_cockpit")
        self._view = "bev"
        self._hex_pts = []
        self._hud = {}
        self._minute = ""

        self.stack = QStackedWidget(self)
        self.bev_widget = bev_widget
        self.stack.addWidget(bev_widget)

        self.lbl_camera = QLabel("Live view")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background: #000; color: #7A93A8;")
        self.lbl_camera.setScaledContents(True)
        self.stack.addWidget(self.lbl_camera)

        self.bezel = HexBezel(self)
        self.hex_stroke = HexStroke(self)

        self.gauge = SpeedGauge(self)
        self.lbl_limit = QLabel("SPEED LIMIT<br><span style='font-size:26px'>89</span><br>LIMIT")
        self.lbl_limit.setAlignment(Qt.AlignCenter)
        self.lbl_limit.setParent(self)
        self.lbl_limit.setTextFormat(Qt.RichText)
        self.lbl_limit.setStyleSheet("color: #D6EEFF; font-size: 11px; font-weight: 700;")

        self.banner_fcw = TriggerBanner("fcw", self)
        self.banner_ldw = TriggerBanner("ldw", self)

        self.settings_wheel = SettingsWheel(self)
        self.settings_wheel.confirmed.connect(self._on_menu_choice)

        self.gear = SettingsGear(self)
        self.gear.clicked.connect(self._toggle_settings)
        self.btn_settings = self.gear

        self.mode_panel = QFrame(self)
        self.mode_panel.hide()
        QVBoxLayout(self.mode_panel)

        self.bottom = QFrame(self)
        bot = QHBoxLayout(self.bottom)
        bot.setContentsMargins(8, 2, 8, 2)
        self.lbl_time = QLabel("◷  —")
        self.lbl_gps = QLabel("GPS")
        self.lbl_status = QLabel("●  ADAS ON")
        self.lbl_lane = QLabel("LANE LOCK")
        self.btn_play = QPushButton("⏸")
        self.btn_open = QPushButton("Open")
        for w in (self.lbl_time, self.lbl_gps, self.lbl_status, self.lbl_lane):
            w.setStyleSheet("color: #8FB8D0; font-size: 11px;")
            bot.addWidget(w)
            if w is not self.lbl_lane:
                bot.addSpacing(18)
        bot.addStretch()
        for b in (self.btn_play, self.btn_open):
            b.setObjectName("ctrl_btn")
            bot.addWidget(b)
        self.bottom.setStyleSheet("background: transparent;")

        self.set_view("bev")

    def sizeHint(self):
        return QSize(WINDOW_W, WINDOW_H)

    def set_view(self, mode):
        self._view = "live" if mode == "live" else "bev"
        self.stack.setCurrentIndex(1 if self._view == "live" else 0)
        if hasattr(self.bev_widget, "setUpdatesEnabled"):
            self.bev_widget.setUpdatesEnabled(self._view == "bev")
        self.view_changed.emit(self._view)

    def _toggle_settings(self):
        if self.settings_wheel.isVisible():
            self.settings_wheel.close_menu()
            self.gear.set_active(False)
        else:
            self.settings_wheel.open_menu()
            self.gear.set_active(True)

    def _on_menu_choice(self, key):
        bev = self.bev_widget
        if key == "live":
            self.set_view("live")
        elif key == "bev":
            self.set_view("bev")
        elif key == "night" and hasattr(bev, "set_env_mode"):
            bev.set_env_mode("night")
        elif key == "grid" and hasattr(bev, "toggle_cinematic_road"):
            bev.toggle_cinematic_road()
        self.settings_wheel.close_menu()
        self.gear.set_active(False)

    def hex_polygon(self):
        return self.bezel.hex_polygon()

    def resizeEvent(self, event):
        self.stack.setGeometry(self.rect())
        self.bezel.setGeometry(self.rect())
        self.hex_stroke.setGeometry(self.rect())
        self.settings_wheel.setGeometry(self.rect())
        self.bezel.raise_()
        w, h = self.width(), self.height()
        cx, cy, rx, ry = _hex_metrics(w, h)
        self._hex_pts = _flat_hex_points(cx, cy, rx, ry)
        self.hex_stroke.set_points(self._hex_pts)
        self.settings_wheel.set_hex(self._hex_pts)

        top_y = cy - ry
        self.gauge.move(
            int(cx - self.gauge.width() / 2),
            int(top_y - 78),
        )

        right = self._hex_pts[0]
        top_right = self._hex_pts[5]
        self.gear.move(
            int(right.x() - self.gear.width() / 2),
            int(right.y() - self.gear.height() / 2),
        )
        self.lbl_limit.adjustSize()
        self.lbl_limit.setFixedWidth(110)
        mx = (top_right.x() + right.x()) / 2.0
        my = (top_right.y() + right.y()) / 2.0
        self.lbl_limit.move(int(mx - 20), int(my - 70))

        left = self._hex_pts[3]
        self.banner_fcw.move(int(left.x() - 10), int(left.y() - self.banner_fcw.height() / 2))
        low_right = self._hex_pts[1]
        lmx = (right.x() + low_right.x()) / 2.0
        lmy = (right.y() + low_right.y()) / 2.0
        self.banner_ldw.move(int(lmx - 40), int(lmy - 20))

        self.bottom.setGeometry(int(cx - 280), int(cy + ry - 36), 560, 30)
        self.hex_stroke.raise_()
        self.settings_wheel.raise_()
        self.gauge.raise_()
        self.lbl_limit.raise_()
        self.gear.raise_()
        self.banner_fcw.raise_()
        self.banner_ldw.raise_()
        self.bottom.raise_()
        super().resizeEvent(event)

    def update_hud(self, *, speed_mps=None, cipo_obj=None, cipo_status="SAFE",
                   alerts=None, fps=None, lane_ok=False):
        alerts = alerts or {}
        kmh = None if speed_mps is None else float(speed_mps) * 3.6
        self.gauge.set_kmh(kmh)

        isa = alerts.get("isa") or {}
        posted = isa.get("posted_mph")
        cand = isa.get("candidate_mph")
        live_limit = posted if posted is not None else cand
        if live_limit is not None:
            n = int(round(float(live_limit)))
            _set_text(
                self.lbl_limit,
                f"SPEED LIMIT<br><span style='font-size:26px;font-weight:800;color:#F4FAFF'>{n}</span><br>LIMIT",
            )
        else:
            n = None
            _set_text(
                self.lbl_limit,
                "SPEED LIMIT<br><span style='font-size:26px;font-weight:800;color:#F4FAFF'>—</span><br>LIMIT",
            )

        dist = None
        if cipo_obj is not None:
            dist = cipo_obj.get("Z_3d") or cipo_obj.get("z")
        if dist is None:
            dist = alerts.get("range_m")
        far_key = None if dist is None else int(round(float(dist)))

        fcw = str(alerts.get("fcw") or "OFF")
        cipo = str(cipo_status or "SAFE")
        fcw_on = fcw in ("FCW", "FCW+") or cipo in ("DANGER", "WARNING")
        ttc = alerts.get("ttc")
        if fcw_on:
            if fcw == "FCW+":
                body = "BRAKE"
                if ttc is not None:
                    body = f"TTC {float(ttc):.1f}s"
            elif far_key is not None:
                body = f"LEAD {far_key} m"
            else:
                body = str(cipo)
            self.banner_fcw.set_trigger(True, "FCW", body)
        else:
            self.banner_fcw.set_trigger(False)

        ldw = str(alerts.get("ldw") or "OFF")
        side = str(alerts.get("ldw_side") or "")
        ldw_on = ldw in ("LEFT", "RIGHT", "LDW") or side in ("LEFT", "RIGHT")
        if ldw_on:
            side_txt = side if side in ("LEFT", "RIGHT") else (ldw if ldw in ("LEFT", "RIGHT") else "")
            self.banner_ldw.set_trigger(True, "LDW", side_txt or "DEPARTURE")
        else:
            self.banner_ldw.set_trigger(False)

        minute = datetime.now().strftime("%H:%M")
        if minute != self._minute:
            self._minute = minute
            _set_text(self.lbl_time, f"◷  {minute}")
        if fps is not None:
            fps_i = int(round(fps))
            if self._hud.get("fps") != fps_i:
                self._hud["fps"] = fps_i
        _set_text(self.lbl_lane, "LANE LOCK" if lane_ok else "LANE …")
