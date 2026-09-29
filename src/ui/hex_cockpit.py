"""Automotive dashboard cockpit over the live ADAS BEV / camera pipeline."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

CYAN = QColor("#3EC8FF")
ICE = QColor("#F4F7FB")
WINDOW_W, WINDOW_H = 1280, 720

VIEWS = (
    ("adas", "ADAS View"),
    ("front", "Front Camera"),
)

_BTN = (
    "QPushButton{background:#121A26;color:#D5DEE8;border:1px solid #2A3848;"
    "border-radius:14px;padding:8px 10px;font-size:11px;font-weight:600;}"
    "QPushButton:hover{border-color:#3EC8FF;}"
    "QPushButton:checked{background:#0E3A62;color:#FFFFFF;border:1px solid #3EC8FF;}"
)


def _set_text(label, text):
    if label.text() != text:
        label.setText(text)


def _stage_rect(w, h):
    # Keep a clean gutter after the left instrument column.
    return QRectF(w * 0.24, h * 0.078, w * 0.54, h * 0.775)


class DonutGauge(QWidget):
    """Open donut, 0–120 km/h, matching the cockpit reference."""

    _VMAX = 120.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kmh = 0.0
        self._shown = 0
        self.setFixedSize(292, 292)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_kmh(self, kmh):
        value = 0.0 if kmh is None else max(0.0, float(kmh))
        shown = int(round(value))
        if shown == self._shown and abs(value - self._kmh) < 0.2:
            return
        self._shown = shown
        self._kmh = value
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = self.width()
        ring = QRectF(32, 32, side - 64, side - 64)
        start = 220 * 16
        sweep = -260 * 16
        p.setPen(QPen(QColor(38, 52, 68), 18, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawArc(ring, start, sweep)
        frac = min(1.0, self._kmh / self._VMAX)
        if frac > 0.01:
            p.setPen(QPen(QColor(40, 150, 255, 70), 26, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(ring, start, int(sweep * frac))
            p.setPen(QPen(CYAN, 16, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(ring, start, int(sweep * frac))
        p.setPen(QColor("#9AABBC"))
        p.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        p.drawText(QRect(8, 218, 54, 24), Qt.AlignCenter, "0")
        p.drawText(QRect(side - 62, 218, 54, 24), Qt.AlignCenter, "120")
        p.setPen(ICE)
        p.setFont(QFont("Segoe UI", 62, QFont.Normal))
        p.drawText(QRect(0, 92, side, 78), Qt.AlignCenter, str(self._shown))
        p.setPen(QColor("#9AABBC"))
        p.setFont(QFont("Segoe UI", 14))
        p.drawText(QRect(0, 168, side, 26), Qt.AlignCenter, "km/h")
        p.end()


class LimitSign(QWidget):
    """Posted speed-limit disc. text() is the live number for HUD checks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = None
        self.setFixedSize(88, 104)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_limit(self, value):
        shown = None if value is None else int(round(float(value)))
        if shown == self._value:
            return
        self._value = shown
        self.update()

    def text(self):
        return "—" if self._value is None else str(self._value)

    def setText(self, text):
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        self.set_limit(int(digits) if digits else None)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        disc = QRectF(9, 3, 70, 70)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#F7F7F7"))
        p.drawEllipse(disc)
        p.setPen(QPen(QColor("#E10600"), 6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(disc.adjusted(3, 3, -3, -3))
        p.setPen(QColor("#111111"))
        p.setFont(QFont("Segoe UI", 20, QFont.Bold))
        p.drawText(disc, Qt.AlignCenter, self.text())
        p.setPen(QColor("#D6DEE8"))
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        p.drawText(QRectF(0, 78, self.width(), 18), Qt.AlignCenter, "Speed Limit")
        p.end()


class TriggerBanner(QWidget):
    """Compact alert chip. Hidden until the tracker fires."""

    def __init__(self, kind="fcw", parent=None):
        super().__init__(parent)
        self._kind = kind
        self._title = "FCW" if kind == "fcw" else "LDW"
        self._body = ""
        self._active = False
        self.setFixedSize(210, 36)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.hide()

    def set_trigger(self, on, title=None, body=""):
        self._active = bool(on)
        if title:
            self._title = title
        self._body = body or ""
        if on:
            if not self.isVisible():
                self.show()
            self.raise_()
            self.update()
        elif self.isVisible():
            self.hide()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        hot = self._kind == "fcw"
        color = QColor("#FF6A3D") if hot else QColor("#FFC14A")
        card = QRectF(1, 1, self.width() - 2, self.height() - 2)
        p.setPen(QPen(color, 1.6))
        p.setBrush(QColor(18, 14, 12, 220))
        p.drawRoundedRect(card, 8, 8)
        p.setPen(ICE)
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        label = self._title if not self._body else f"{self._title}  {self._body}"
        p.drawText(card, Qt.AlignCenter, label)
        p.end()


class ObjectOverlay(QWidget):
    """Lightweight projected object boxes over the ADAS chase view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._objects = []
        self._stage = QRectF()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_stage(self, rect):
        self._stage = QRectF(rect)
        self.update()

    def set_objects(self, objects):
        rows = []
        for obj in objects or []:
            try:
                z = float(obj.get("Z_3d", 0.0))
                x = float(obj.get("X_3d", 0.0))
            except (TypeError, ValueError):
                continue
            if 4.0 <= z <= 75.0 and abs(x) <= 8.0:
                rows.append((z, x, bool(obj.get("is_cipo"))))
        self._objects = sorted(rows)[:3]
        self.update()

    def paintEvent(self, event):
        if self._stage.isEmpty():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        stage = self._stage
        for z, x, is_cipo in self._objects:
            depth = min(1.0, max(0.0, (z - 4.0) / 71.0))
            perspective = depth ** 0.62
            cy = stage.bottom() - stage.height() * (0.20 + 0.62 * perspective)
            lateral_scale = stage.width() * (0.070 - 0.035 * depth)
            cx = stage.center().x() + x * lateral_scale
            box_w = max(34.0, 92.0 - z * 0.75)
            box_h = box_w * 0.62
            rect = QRectF(cx - box_w / 2, cy - box_h / 2, box_w, box_h)
            color = QColor("#31E6A1") if not is_cipo else QColor("#52F2B2")
            p.setPen(QPen(QColor(49, 230, 161, 55), 7))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect, 4, 4)
            p.setPen(QPen(color, 1.8))
            p.drawRoundedRect(rect, 4, 4)
            label = QRectF(rect.left(), rect.bottom() + 4, rect.width(), 20)
            p.setPen(QColor("#EAF7F2"))
            p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
            p.drawText(label, Qt.AlignCenter, f"{int(round(z))} m")
        p.end()


class SettingsButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("⚙   Settings", parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(148, 52)
        self._on = False
        self._apply()

    def set_active(self, on):
        self._on = bool(on)
        self._apply()

    def _apply(self):
        border = "#3EC8FF" if self._on else "#314154"
        bg = "#12324E" if self._on else "#121A26"
        self.setStyleSheet(
            "QPushButton{background:%s;color:#F2F6FB;border:1px solid %s;"
            "border-radius:16px;font-size:13px;font-weight:700;}"
            "QPushButton:hover{border-color:#3EC8FF;}" % (bg, border)
        )


class SettingsSheet(QFrame):
    """View picker opened by the Settings button."""

    confirmed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settings_sheet")
        self.setStyleSheet(
            "QFrame#settings_sheet{background:rgba(14,20,30,242);"
            "border:1px solid rgba(120,170,210,80);border-radius:18px;}"
            "QLabel{color:#E8EEF6;background:transparent;}"
            + _BTN
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel("View Mode")
        title.setStyleSheet("font-size:16px;font-weight:700;color:#F4F7FB;")
        close = QPushButton("✕")
        close.setFixedSize(28, 28)
        close.setStyleSheet(
            "QPushButton{background:transparent;color:#C5D0DC;border:none;font-size:14px;}"
        )
        close.clicked.connect(self.close_menu)
        head.addWidget(title)
        head.addStretch()
        head.addWidget(close)
        root.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.view_buttons = {}
        for key, label in VIEWS:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(64)
            button.clicked.connect(lambda _=False, k=key: self.confirmed.emit(k))
            row.addWidget(button)
            self.view_buttons[key] = button
        root.addLayout(row)

        more = QLabel("More Settings")
        more.setStyleSheet("color:#9AABBC;font-size:12px;font-weight:600;")
        root.addWidget(more)
        extra = QHBoxLayout()
        for key, label in (("night", "Night Mode"), ("grid", "Grid HUD"), ("full", "Full View")):
            button = QPushButton(label)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=key: self.confirmed.emit(k))
            extra.addWidget(button)
        root.addLayout(extra)

        self.tools = QHBoxLayout()
        root.addLayout(self.tools)
        self.mode_slot = QVBoxLayout()
        root.addLayout(self.mode_slot)
        self.hide()

    def open_menu(self):
        self.show()
        self.raise_()

    def close_menu(self):
        self.hide()


class HexCockpit(QWidget):
    view_changed = Signal(str)

    def __init__(self, bev_widget, parent=None):
        super().__init__(parent)
        self.setObjectName("hex_cockpit")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#hex_cockpit{background:#05070C;}")
        self._view = "bev"
        self._mode = "adas"
        self._full_view = False
        self._minute = ""
        self._fps = None

        self.stack = QStackedWidget(self)
        self.bev_widget = bev_widget
        self.stack.addWidget(bev_widget)
        self.lbl_camera = QLabel("Live view")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background:#000;color:#7A93A8;")
        self.lbl_camera.setScaledContents(True)
        self.stack.addWidget(self.lbl_camera)

        self.gauge = DonutGauge(self)
        self.lbl_limit = LimitSign(self)
        self.banner_fcw = TriggerBanner("fcw", self)
        self.banner_ldw = TriggerBanner("ldw", self)

        self.topbar = QFrame(self)
        self.topbar.setStyleSheet("background:transparent;")
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(0, 0, 0, 0)
        top.addStretch()
        self.lbl_range = QLabel("▣  Range —")
        self.lbl_clock = QLabel("—")
        self.lbl_date = QLabel("—")
        self.lbl_temp = QLabel("28°C")
        sep = QLabel("|")
        for label in (self.lbl_range, self.lbl_clock, sep, self.lbl_date, self.lbl_temp):
            label.setStyleSheet("color:#D5DEE8;font-size:14px;font-weight:600;background:transparent;")
        self.lbl_clock.setMinimumWidth(120)
        self.lbl_date.setMinimumWidth(150)
        self.lbl_clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_date.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top.addWidget(self.lbl_range)
        top.addSpacing(22)
        top.addWidget(self.lbl_clock)
        top.addSpacing(16)
        top.addWidget(sep)
        top.addSpacing(16)
        top.addWidget(self.lbl_date)
        top.addSpacing(22)
        top.addWidget(self.lbl_temp)
        top.addStretch()
        self.lbl_fps = QLabel("FPS —")
        self.lbl_fps.setStyleSheet("color:#C5D0DC;font-size:13px;font-weight:600;background:transparent;")

        self.status_card = QFrame(self)
        self.status_card.setStyleSheet(
            "QFrame{background:rgba(10,16,26,210);border:1px solid #243244;border-radius:18px;}"
            "QLabel{background:transparent;border:none;}"
        )
        card = QVBoxLayout(self.status_card)
        card.setContentsMargins(16, 14, 16, 16)
        card.setSpacing(8)
        self.lbl_drive = QLabel("D")
        self.lbl_drive.setAlignment(Qt.AlignCenter)
        self.lbl_drive.setStyleSheet("color:#F4F7FB;font-size:30px;font-weight:600;")
        self.lbl_drive_sub = QLabel("NORMAL")
        self.lbl_drive_sub.setAlignment(Qt.AlignCenter)
        self.lbl_drive_sub.setStyleSheet("color:#8FA0B3;font-size:11px;font-weight:700;")
        card.addWidget(self.lbl_drive)
        card.addWidget(self.lbl_drive_sub)
        card.addSpacing(8)

        self.dot_lane = QLabel("●")
        self.dot_fcw = QLabel("●")
        self.dot_sign = QLabel("●")
        self.dot_obj = QLabel("●")
        self._status_rows = (
            (self.dot_lane, "╱╲   Lane Keeping"),
            (self.dot_fcw, "▣   Forward Collision"),
            (self.dot_sign, "◉   Traffic Sign"),
            (self.dot_obj, "▤   Object Detection"),
        )
        self._name_labels = []
        for dot, name in self._status_rows:
            row = QHBoxLayout()
            label = QLabel(name)
            label.setStyleSheet("color:#E6EDF5;font-size:13px;font-weight:600;")
            dot.setStyleSheet("color:#5C6B7A;font-size:16px;")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(dot)
            card.addLayout(row)
            self._name_labels.append((dot, label))

        self.settings_wheel = SettingsSheet(self)
        self.settings_wheel.confirmed.connect(self._on_menu_choice)
        self.gear = SettingsButton(self)
        self.gear.clicked.connect(self._toggle_settings)
        self.btn_settings = self.gear

        self.mode_panel = QFrame(self.settings_wheel)
        self.settings_wheel.mode_slot.addWidget(self.mode_panel)
        QVBoxLayout(self.mode_panel)

        self.btn_play = QPushButton("Pause")
        self.btn_open = QPushButton("Open")
        self.btn_full = self.btn_play
        for button in (self.btn_play, self.btn_open):
            button.setCursor(Qt.PointingHandCursor)
            self.settings_wheel.tools.addWidget(button)

        self._view_btns = {}
        self.bottom = QFrame(self)
        self.bottom.setStyleSheet("background:transparent;")
        bot = QHBoxLayout(self.bottom)
        bot.setContentsMargins(8, 0, 8, 0)
        bot.setSpacing(8)
        bot.addWidget(self.lbl_fps)
        bot.addStretch()
        for key, label in VIEWS:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumSize(180, 52)
            button.setStyleSheet(_BTN)
            button.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            bot.addWidget(button)
            self._view_btns[key] = button
        bot.addStretch()
        # Settings remain available programmatically, but the dashboard button
        # is intentionally omitted to keep the lower-right area uncluttered.
        self.gear.hide()

        self.btn_restore = QPushButton("Restore", self)
        self.btn_restore.setFixedSize(92, 34)
        self.btn_restore.setCursor(Qt.PointingHandCursor)
        self.btn_restore.setStyleSheet(
            "QPushButton{background:rgba(8,16,28,210);color:#E8F4FF;"
            "border:1px solid #3EC8FF;border-radius:10px;font-weight:700;}"
        )
        self.btn_restore.clicked.connect(self.toggle_interface)
        self.btn_restore.hide()

        self._paint_status_dots(False, False, False, False)
        self.set_mode("adas")

    def sizeHint(self):
        return QSize(WINDOW_W, WINDOW_H)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#05070C"))
        inner = QRectF(8, 6, self.width() - 16, self.height() - 12)
        p.setPen(QPen(QColor("#243140"), 1.4))
        p.setBrush(QColor("#0B1018"))
        p.drawRoundedRect(inner, 26, 26)
        p.end()

    def set_view(self, mode):
        self._view = "live" if mode == "live" else "bev"
        self.stack.setCurrentIndex(1 if self._view == "live" else 0)
        if hasattr(self.bev_widget, "setUpdatesEnabled"):
            self.bev_widget.setUpdatesEnabled(self._view == "bev")
        self.view_changed.emit(self._view)

    def set_mode(self, mode):
        self._mode = mode if mode in dict(VIEWS) else "adas"
        if self._mode in ("live", "front"):
            self.set_view("live")
        else:
            self.set_view("bev")
            if hasattr(self.bev_widget, "set_dashboard_camera"):
                self.bev_widget.set_dashboard_camera("bird" if self._mode == "bird" else "chase")
        for group in (self._view_btns, self.settings_wheel.view_buttons):
            for key, button in group.items():
                button.setChecked(key == self._mode)

    def _toggle_settings(self):
        if self.settings_wheel.isVisible():
            self.settings_wheel.close_menu()
            self.gear.set_active(False)
        else:
            self.settings_wheel.open_menu()
            self.gear.set_active(True)
            self.settings_wheel.raise_()

    def _on_menu_choice(self, key):
        bev = self.bev_widget
        if key in dict(VIEWS):
            self.set_mode(key)
            self.settings_wheel.close_menu()
            self.gear.set_active(False)
            return
        if key == "live":
            self.set_mode("live")
        elif key == "bev":
            self.set_mode("adas")
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
        self._full_view = not bool(visible)
        self.settings_wheel.close_menu()
        self.gear.set_active(False)
        chrome = (
            self.gauge, self.lbl_limit, self.bottom, self.topbar,
            self.status_card, self.lbl_fps,
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
            stage = _stage_rect(self.width(), self.height())
            self.stack.setMask(QRegion(self._stage_path(stage).toFillPolygon().toPolygon()))
            if self.banner_fcw._active:
                self.banner_fcw.show()
            if self.banner_ldw._active:
                self.banner_ldw.show()
            self.btn_restore.hide()
            self._raise_chrome()

    def _stage_path(self, rect):
        path = QPainterPath()
        path.addRoundedRect(rect, 18, 18)
        return path

    def _raise_chrome(self):
        for widget in (
            self.gauge, self.lbl_limit, self.topbar, self.status_card, self.bottom,
            self.banner_fcw, self.banner_ldw, self.settings_wheel,
        ):
            widget.raise_()

    def _paint_status_dots(self, lane_ok, fcw_on, sign_ok, detect_ok):
        states = (lane_ok, not fcw_on, sign_ok, detect_ok)
        for dot, on in zip((self.dot_lane, self.dot_fcw, self.dot_sign, self.dot_obj), states):
            color = "#3DDC97" if on else "#FF5A3C" if dot is self.dot_fcw and fcw_on else "#5C6B7A"
            if dot is self.dot_fcw and fcw_on:
                color = "#FF5A3C"
            elif on:
                color = "#3DDC97"
            else:
                color = "#5C6B7A"
            dot.setStyleSheet(f"color:{color};font-size:14px;background:transparent;")

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        self.stack.setGeometry(self.rect())
        stage = _stage_rect(w, h)
        if self._full_view:
            self.stack.clearMask()
        else:
            self.stack.setMask(QRegion(self._stage_path(stage).toFillPolygon().toPolygon()))

        self.topbar.setGeometry(int(w * 0.28), 10, int(w * 0.44), 30)
        self.gauge.move(2, int(h * 0.13))
        self.lbl_limit.move(
            int(self.gauge.x() + (self.gauge.width() - self.lbl_limit.width()) / 2),
            int(self.gauge.geometry().bottom() + 4),
        )

        panel_x = int(w * 0.785)
        self.status_card.setGeometry(panel_x, 52, w - panel_x - 18, 280)
        self.banner_fcw.move(panel_x, 344)
        self.banner_ldw.move(panel_x, 386)
        self.bottom.setGeometry(24, int(h * 0.855), w - 48, 64)
        self.settings_wheel.setGeometry(int(w * 0.515), int(h * 0.42), int(w * 0.465), 300)
        self.btn_restore.move(w - 110, 16)
        if not self._full_view:
            self._raise_chrome()
        else:
            self.stack.raise_()
            self.btn_restore.raise_()
        super().resizeEvent(event)

    def update_hud(self, *, speed_mps=None, cipo_obj=None, cipo_status="SAFE",
                   alerts=None, fps=None, lane_ok=False, objects=None):
        alerts = alerts or {}
        kmh = None if speed_mps is None else float(speed_mps) * 3.6
        self.gauge.set_kmh(kmh)

        isa = alerts.get("isa") or {}
        posted = isa.get("posted_mph")
        cand = isa.get("candidate_mph")
        live_limit = posted if posted is not None else cand
        self.lbl_limit.set_limit(live_limit)

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
                body = f"TTC {float(ttc):.1f}s" if ttc is not None else "BRAKE"
            else:
                body = f"LEAD {far_key} m" if far_key is not None else "FORWARD"
            self.banner_fcw.set_trigger(True, "FCW", body)
        else:
            self.banner_fcw.set_trigger(False)

        ldw = str(alerts.get("ldw") or "OFF")
        side = str(alerts.get("ldw_side") or "")
        ldw_on = ldw in ("LEFT", "RIGHT") or side in ("LEFT", "RIGHT")
        if ldw_on:
            side_txt = side if side in ("LEFT", "RIGHT") else ldw
            self.banner_ldw.set_trigger(True, "LDW", side_txt)
        else:
            self.banner_ldw.set_trigger(False)

        if self._full_view:
            self.banner_fcw.hide()
            self.banner_ldw.hide()

        now = datetime.now()
        minute = now.strftime("%I:%M %p").lstrip("0")
        if minute != self._minute:
            self._minute = minute
            _set_text(self.lbl_clock, minute)
            _set_text(self.lbl_date, now.strftime("%a, %b %d"))
        if fps is not None:
            fps_i = int(round(fps))
            if fps_i != self._fps:
                self._fps = fps_i
                _set_text(self.lbl_fps, f"FPS {fps_i}")
        self._paint_status_dots(bool(lane_ok), fcw_on, live_limit is not None, True)
