"""Curved dual-wing infotainment shell over the live ADAS BEV/camera pipeline."""

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
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
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
WINDOW_W, WINDOW_H = 1280, 720


def _viewport_rect(w, h):
    return QRectF(7, 7, w - 14, h - 14)


def _viewport_path(w, h):
    path = QPainterPath()
    path.addRoundedRect(_viewport_rect(w, h), 18, 18)
    return path


def _wing_edge(w, h, right=False):
    x = w * (0.76 if right else 0.24)
    sign = -1.0 if right else 1.0
    path = QPainterPath(QPoint(int(x), int(h * 0.18)))
    path.cubicTo(
        x + sign * w * 0.065, h * 0.30,
        x + sign * w * 0.065, h * 0.68,
        x, h * 0.80,
    )
    return path


def _wing_path(w, h, right=False):
    edge = _wing_edge(w, h, right)
    if right:
        path = QPainterPath(QPoint(int(w * 0.975), int(h * 0.07)))
        path.lineTo(w * 0.76, h * 0.18)
        path.connectPath(edge)
        path.lineTo(w * 0.975, h * 0.90)
    else:
        path = QPainterPath(QPoint(int(w * 0.025), int(h * 0.07)))
        path.lineTo(w * 0.24, h * 0.18)
        path.connectPath(edge)
        path.lineTo(w * 0.025, h * 0.90)
    path.closeSubpath()
    return path


def _wing_border(w, h, right=False):
    """Open inner border: upper sweep, concave edge, and lower sweep."""
    if right:
        path = QPainterPath(QPoint(int(w * 0.975), int(h * 0.07)))
        path.lineTo(w * 0.76, h * 0.18)
    else:
        path = QPainterPath(QPoint(int(w * 0.025), int(h * 0.07)))
        path.lineTo(w * 0.24, h * 0.18)
    path.connectPath(_wing_edge(w, h, right))
    path.lineTo(w * (0.975 if right else 0.025), h * 0.90)
    return path


def _center_path(w, h):
    """Curved center opening used as the real BEV/live-video mask."""
    left_x, right_x = w * 0.24, w * 0.76
    top_y, bottom_y = h * 0.18, h * 0.80
    path = QPainterPath(QPoint(int(left_x), int(top_y)))
    # Upper boundary joins both wing edges without a gap.
    path.cubicTo(w * 0.38, h * 0.10, w * 0.62, h * 0.10, right_x, top_y)
    # Right concave wing edge.
    path.cubicTo(
        right_x - w * 0.065, h * 0.30,
        right_x - w * 0.065, h * 0.68,
        right_x, bottom_y,
    )
    # Lower boundary joins both wing edges without a gap.
    path.cubicTo(w * 0.62, h * 0.88, w * 0.38, h * 0.88, left_x, bottom_y)
    # Left concave wing edge, traversed upward.
    path.cubicTo(
        left_x + w * 0.065, h * 0.68,
        left_x + w * 0.065, h * 0.30,
        left_x, top_y,
    )
    path.closeSubpath()
    return path


def _set_text(label, text):
    if label.text() != text:
        label.setText(text)


class HexBezel(QWidget):
    """Cached curved shell and clipped honeycomb wings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._cache = None

    def hex_polygon(self):
        return QPolygon(_viewport_path(self.width(), self.height()).toFillPolygon().toPolygon())

    def resizeEvent(self, event):
        self._rebuild()
        super().resizeEvent(event)

    def _rebuild(self):
        w, h = self.width(), self.height()
        if w < 8 or h < 8:
            return
        self.clearMask()
        self._cache = QPixmap(w, h)
        self._cache.fill(Qt.transparent)
        p = QPainter(self._cache)
        p.setRenderHint(QPainter.Antialiasing, True)

        for right in (False, True):
            wing = _wing_path(w, h, right)
            p.save()
            p.setClipPath(wing)
            p.fillPath(wing, QColor("#071426"))
            self._paint_honeycomb(p, w, h, right)
            p.restore()

        outer = QRectF(6, 6, w - 12, h - 12)
        for width, alpha in ((7, 16), (1.6, 85)):
            glow = QColor("#244D72")
            glow.setAlpha(alpha)
            p.setPen(QPen(glow, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(outer, 18, 18)

        self._draw_dark_rail(p, w, h, top=True)
        self._draw_dark_rail(p, w, h, top=False)
        p.end()
        self.update()

    def _draw_dark_rail(self, p, w, h, top):
        """Center boundary joins the left and right wing corners exactly."""
        y = h * (0.18 if top else 0.80)
        ctrl = h * (0.10 if top else 0.88)
        path = QPainterPath(QPoint(int(w * 0.24), int(y)))
        path.cubicTo(w * 0.38, ctrl, w * 0.62, ctrl, w * 0.76, y)
        p.setPen(QPen(QColor(20, 64, 92, 175), 1.25, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def paintEvent(self, event):
        if self._cache is None:
            return
        QPainter(self).drawPixmap(0, 0, self._cache)

    def _paint_honeycomb(self, painter, w, h, right=False):
        r = 18.0
        dx = r * math.sqrt(3)
        dy = r * 1.5
        # Cells brighten smoothly toward each wing's inner curved edge.
        if right:
            gradient = QLinearGradient(w, 0, w * 0.74, 0)
        else:
            gradient = QLinearGradient(0, 0, w * 0.26, 0)
        gradient.setColorAt(0.0, QColor(38, 88, 128, 48))
        gradient.setColorAt(0.68, QColor(55, 125, 174, 105))
        gradient.setColorAt(1.0, QColor(92, 185, 224, 175))
        painter.setPen(QPen(QBrush(gradient), 1.15))
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
    """Unmasked cyan inner wing edges over the live viewport."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._size = QSize()

    def set_points(self, pts):
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for right in (False, True):
            path = _wing_border(self.width(), self.height(), right)
            for width, alpha in ((10, 24), (2.2, 215)):
                glow = QColor(CYAN)
                glow.setAlpha(alpha)
                p.setPen(QPen(glow, width, Qt.SolidLine, Qt.RoundCap))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
        p.end()


class SpeedGauge(QWidget):
    """Reference-style full donut gauge."""

    _VMAX = 160.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kmh = 0.0
        self._shown = 0
        self.setFixedSize(220, 220)
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
        cx = cy = self.width() / 2.0
        r = 78.0
        ring = QRectF(cx - r, cy - r, r * 2, r * 2)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(25, 72, 120, 200), 14, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(ring)

        for i in range(8):
            deg = i * 45.0
            rad = math.radians(deg)
            r0, r1 = r + 15, r + 23
            x0 = cx + r0 * math.cos(rad)
            y0 = cy - r0 * math.sin(rad)
            x1 = cx + r1 * math.cos(rad)
            y1 = cy - r1 * math.sin(rad)
            p.setPen(QPen(QColor(175, 220, 245, 170), 2))
            p.drawLine(QPoint(int(x0), int(y0)), QPoint(int(x1), int(y1)))

        frac = min(1.0, self._kmh / self._VMAX)
        if frac > 0.01:
            fill_span = int(-360 * frac * 16)
            p.setPen(QPen(QColor(40, 160, 255, 75), 22, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(ring, 90 * 16, fill_span)
            p.setPen(QPen(CYAN, 13, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(ring, 90 * 16, fill_span)

        p.setPen(ICE)
        p.setFont(QFont("Segoe UI", 46, QFont.Normal))
        p.drawText(QRect(0, 66, self.width(), 60), Qt.AlignCenter, f"{self._shown}")
        p.setPen(QColor("#9EC8E0"))
        p.setFont(QFont("Segoe UI", 13))
        p.drawText(QRect(0, 126, self.width(), 24), Qt.AlignCenter, "km/h")
        p.end()


class TriggerBanner(QWidget):
    """Edge HUD chip: FCW (left) or LDW (right). Hidden until the tracker fires."""

    def __init__(self, kind="fcw", parent=None):
        super().__init__(parent)
        self._kind = kind
        self._title = "FCW" if kind == "fcw" else "LDW"
        self._body = ""
        self._active = False
        self._pulse = 0.0
        self.setFixedSize(210, 94)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def _tick(self):
        self._pulse = (self._pulse + 0.12) % (2.0 * math.pi)
        self.update()

    def set_trigger(self, on, title=None, body=""):
        self._active = bool(on)
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
        hot = self._kind == "fcw"
        border = QColor("#FF9348")
        card = QRectF(7, 7, self.width() - 14, self.height() - 14)
        path = QPainterPath()
        path.addRoundedRect(card, 13, 13)
        pulse = int(22 + 18 * abs(math.sin(self._pulse)))
        p.setPen(QPen(QColor(255, 120, 50, pulse), 9))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.setPen(QPen(border, 2.2))
        p.setBrush(QColor(18, 15, 20, 220))
        p.drawPath(path)
        if hot:
            shield = QPainterPath(QPoint(38, 24))
            shield.lineTo(56, 31)
            shield.cubicTo(55, 57, 48, 67, 38, 73)
            shield.cubicTo(28, 67, 21, 57, 20, 31)
            shield.closeSubpath()
            p.setPen(QPen(border, 3))
            p.setBrush(Qt.NoBrush)
            p.drawPath(shield)
            p.drawLine(QPoint(29, 47), QPoint(36, 54))
            p.drawLine(QPoint(36, 54), QPoint(48, 40))
            p.setPen(ICE)
            p.setFont(QFont("Segoe UI", 18, QFont.Bold))
            p.drawText(QRect(70, 20, 130, 34), Qt.AlignLeft | Qt.AlignVCenter, self._title)
            p.setPen(QColor("#D6D5D8"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(QRect(70, 51, 130, 22), Qt.AlignLeft | Qt.AlignVCenter, self._body)
        else:
            text = f"{self._title} {self._body}".strip()
            p.setPen(ICE)
            p.setFont(QFont("Segoe UI", 17, QFont.Bold))
            p.drawText(card, Qt.AlignCenter, text)
        p.end()


class SettingsGear(QWidget):
    """Icon-only 2x2 feature launcher."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(62, 62)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._angle = 0.0
        self._fast = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_active(self, on):
        self._fast = bool(on)
        self.update()

    def _tick(self):
        self._angle = (self._angle + 1.0) % 360.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tile = QRectF(5, 5, 52, 52)
        p.setPen(QPen(CYAN, 2))
        p.setBrush(QColor(7, 24, 43, 225 if self._fast else 185))
        p.drawRoundedRect(tile, 15, 15)
        p.setPen(QPen(ICE if self._fast else CYAN, 1.8))
        p.setBrush(Qt.NoBrush)
        size, gap = 10.0, 7.0
        x0 = y0 = 5 + (52 - (size * 2 + gap)) / 2
        for row in range(2):
            for col in range(2):
                p.drawRoundedRect(
                    QRectF(x0 + col * (size + gap), y0 + row * (size + gap), size, size),
                    2, 2,
                )
        p.end()


class SettingsWheel(QWidget):
    """Frosted cards that rise from the bottom; scroll to choose, tap to confirm."""

    confirmed = Signal(str)

    ITEMS = (
        ("night", "NIGHT MODE"),
        ("grid", "GRID HUD"),
        ("bev", "ADAS BEV"),
        ("live", "LIVE VIEW"),
        ("full", "FULL VIEW"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._index = 2
        self._rise = 0.0
        self._clip = QPainterPath()
        self._anim = None
        self.hide()

    def _get_rise(self):
        return self._rise

    def _set_rise(self, v):
        self._rise = float(v)
        self.update()

    rise = Property(float, _get_rise, _set_rise)

    def set_hex(self, pts):
        if isinstance(pts, QPainterPath):
            self._clip = QPainterPath(pts)
            return
        self._clip = QPainterPath()
        if pts:
            polygon = QPolygon([QPoint(int(p.x()), int(p.y())) for p in pts])
            self._clip.addPolygon(polygon)

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
        if not self._clip.isEmpty():
            p.setClipPath(self._clip)
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
        y = cy + rel * 56.0 + (1.0 - t) * (220.0 + abs(rel) * 36.0)
        focus = rel == 0
        scale = 0.82 + 0.18 * t
        if rel < 0:
            w, h = (270.0 + (22 if rel == -1 else 0)) * scale, 45.0 * scale
        elif rel > 0:
            w, h = 430.0 * scale, 72.0 * scale
            y += 8
        else:
            w, h = 370.0 * scale, 82.0 * scale
        alpha_fill = int((112 if focus else 58) * t)
        alpha_border = int((245 if focus else 125) * t)
        slant = (14.0 if rel >= 0 else 8.0) * scale
        poly = QPolygon(
            [
                QPoint(int(cx - w / 2 + slant), int(y - h / 2)),
                QPoint(int(cx + w / 2 - slant), int(y - h / 2)),
                QPoint(int(cx + w / 2), int(y + h / 2)),
                QPoint(int(cx - w / 2), int(y + h / 2)),
            ]
        )
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
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
        rect = poly.boundingRect()
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
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#hex_cockpit{background:#02050C;}")
        self._view = "bev"
        self._full_view = False
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
        self.lbl_limit = QLabel("SPEED LIMIT<br><span style='font-size:25px'>—</span>")
        self.lbl_limit.setAlignment(Qt.AlignCenter)
        self.lbl_limit.setParent(self)
        self.lbl_limit.setTextFormat(Qt.RichText)
        self.lbl_limit.setFixedSize(114, 62)
        self.lbl_limit.setStyleSheet(
            "QLabel{color:#9FC9E3; font-size:10px; font-weight:700;"
            "background:rgba(8,27,47,185); border:1px solid rgba(62,200,255,145);"
            "border-radius:12px; padding-top:3px;}"
        )

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
        bot.setContentsMargins(4, 0, 8, 0)
        self.lbl_time = QLabel("◷  —")
        self.lbl_gps = QLabel("GPS")
        self.lbl_status = QLabel("●  ADAS ON")
        self.lbl_lane = QLabel("♙  LANE LOCK")
        self.lbl_fps = QLabel("FPS —")
        self.btn_play = QPushButton("⏸")
        self.btn_open = QPushButton("↗")
        self.btn_full = QPushButton("⛶")
        bot.addWidget(self.gear)
        bot.addStretch()
        chips = (self.lbl_time, self.lbl_gps, self.lbl_status, self.lbl_lane, self.lbl_fps)
        for i, w in enumerate(chips):
            w.setStyleSheet("color: #90A9BA; font-size: 11px; font-weight: 600;")
            bot.addWidget(w)
            if i < len(chips) - 1:
                sep = QLabel("│")
                sep.setStyleSheet("color: rgba(100,145,175,85); font-size: 11px;")
                bot.addSpacing(14)
                bot.addWidget(sep)
                bot.addSpacing(14)
        bot.addStretch()
        for b in (self.btn_play, self.btn_open, self.btn_full):
            b.setObjectName("ctrl_btn")
            b.setFixedSize(28, 22)
            b.setStyleSheet(
                "QPushButton{background:rgba(7,20,35,150);color:#8FB8D0;"
                "border:1px solid rgba(62,200,255,80);border-radius:5px;font-size:10px;}"
            )
            bot.addWidget(b)
            bot.addSpacing(5)
        self.bottom.setStyleSheet("background: transparent;")
        self.btn_full.setToolTip("Hide interface and expand the current view")
        self.btn_full.clicked.connect(self.toggle_interface)

        self.btn_restore = QPushButton("◱", self)
        self.btn_restore.setFixedSize(40, 32)
        self.btn_restore.setToolTip("Restore interface")
        self.btn_restore.setStyleSheet(
            "QPushButton{background:rgba(4,14,26,190);color:#BFEAFF;"
            "border:1px solid rgba(62,200,255,150);border-radius:8px;font-size:17px;}"
            "QPushButton:hover{border-color:#3EC8FF;background:rgba(8,28,48,225);}"
        )
        self.btn_restore.clicked.connect(self.toggle_interface)
        self.btn_restore.hide()

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
        elif key == "full":
            self.set_interface_visible(False)
        self.settings_wheel.close_menu()
        self.gear.set_active(False)

    def toggle_interface(self):
        self.set_interface_visible(self._full_view)

    def set_interface_visible(self, visible):
        """Show the cockpit chrome or expand the active BEV/live view to the full display."""
        self._full_view = not bool(visible)
        self.settings_wheel.close_menu()
        self.gear.set_active(False)
        chrome = (
            self.bezel,
            self.hex_stroke,
            self.gauge,
            self.lbl_limit,
            self.gear,
            self.bottom,
        )
        if self._full_view:
            for widget in chrome:
                widget.hide()
            self.banner_fcw.hide()
            self.banner_ldw.hide()
            self.stack.clearMask()
            self.stack.raise_()
            self.btn_restore.show()
            self.btn_restore.raise_()
        else:
            for widget in chrome:
                widget.show()
            viewport = _center_path(self.width(), self.height())
            self.stack.setMask(QRegion(viewport.toFillPolygon().toPolygon()))
            if self.banner_fcw._active:
                self.banner_fcw.show()
            if self.banner_ldw._active:
                self.banner_ldw.show()
            self.btn_restore.hide()
            self.bezel.raise_()
            self.hex_stroke.raise_()
            self.gauge.raise_()
            self.lbl_limit.raise_()
            self.gear.raise_()
            self.banner_fcw.raise_()
            self.banner_ldw.raise_()
            self.bottom.raise_()
        self.update()

    def hex_polygon(self):
        return self.bezel.hex_polygon()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        self.stack.setGeometry(self.rect())
        viewport = _center_path(w, h)
        if self._full_view:
            self.stack.clearMask()
        else:
            self.stack.setMask(QRegion(viewport.toFillPolygon().toPolygon()))
        self.bezel.setGeometry(self.rect())
        self.hex_stroke.setGeometry(self.rect())
        self.settings_wheel.setGeometry(self.rect())
        self.bezel.raise_()
        self._hex_pts = list(viewport.toFillPolygon())
        self.hex_stroke.set_points(self._hex_pts)
        self.settings_wheel.set_hex(viewport)

        lx, ly = int(w * 0.145), int(h * 0.40)
        self.gauge.move(lx - self.gauge.width() // 2, ly - self.gauge.height() // 2)
        self.lbl_limit.move(lx - self.lbl_limit.width() // 2, int(h * 0.66))

        rx = int(w * 0.865)
        self.banner_fcw.move(rx - self.banner_fcw.width() // 2, int(h * 0.29))
        self.banner_ldw.move(rx - self.banner_ldw.width() // 2, int(h * 0.54))

        self.bottom.setGeometry(int(w * 0.245), int(h * 0.842), int(w * 0.51), 62)
        self.btn_restore.move(w - self.btn_restore.width() - 18, 18)
        self.hex_stroke.raise_()
        self.settings_wheel.raise_()
        self.gauge.raise_()
        self.lbl_limit.raise_()
        self.gear.raise_()
        self.banner_fcw.raise_()
        self.banner_ldw.raise_()
        self.bottom.raise_()
        if self._full_view:
            self.stack.raise_()
            self.btn_restore.raise_()
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
                f"SPEED LIMIT<br><span style='font-size:25px;font-weight:800;color:#F4FAFF'>{n}</span>",
            )
        else:
            n = None
            _set_text(
                self.lbl_limit,
                "SPEED LIMIT<br><span style='font-size:25px;font-weight:800;color:#F4FAFF'>—</span>",
            )

        dist = None
        if cipo_obj is not None:
            dist = cipo_obj.get("Z_3d") or cipo_obj.get("z")
        if dist is None:
            dist = alerts.get("range_m")
        far_key = None if dist is None else int(round(float(dist)))

        fcw = str(alerts.get("fcw") or "OFF")
        fcw_on = fcw in ("FCW", "FCW+")
        ttc = alerts.get("ttc")
        if fcw_on:
            if fcw == "FCW+":
                body = "BRAKE"
                if ttc is not None:
                    body = f"TTC {float(ttc):.1f}s"
            elif far_key is not None:
                body = f"LEAD {far_key} m"
            else:
                body = "FORWARD"
            self.banner_fcw.set_trigger(True, "FCW", body)
        else:
            self.banner_fcw.set_trigger(False)

        ldw = str(alerts.get("ldw") or "OFF")
        side = str(alerts.get("ldw_side") or "")
        ldw_on = ldw in ("LEFT", "RIGHT") or side in ("LEFT", "RIGHT")
        if ldw_on:
            side_txt = side if side in ("LEFT", "RIGHT") else (ldw if ldw in ("LEFT", "RIGHT") else "")
            self.banner_ldw.set_trigger(True, "LDW", side_txt or "DEPARTURE")
        else:
            self.banner_ldw.set_trigger(False)

        if self._full_view:
            self.banner_fcw.hide()
            self.banner_ldw.hide()

        minute = datetime.now().strftime("%H:%M")
        if minute != self._minute:
            self._minute = minute
            _set_text(self.lbl_time, f"◷  {minute}")
        if fps is not None:
            fps_i = int(round(fps))
            if self._hud.get("fps") != fps_i:
                self._hud["fps"] = fps_i
                _set_text(self.lbl_fps, f"FPS {fps_i}")
        elif self._hud.get("fps") is not None:
            self._hud["fps"] = None
            _set_text(self.lbl_fps, "FPS —")
        _set_text(self.lbl_lane, "♙  LANE LOCK" if lane_ok else "♙  LANE …")
