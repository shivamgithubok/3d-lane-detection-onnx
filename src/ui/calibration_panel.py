"""Compact extrinsics bar: pitch + height on one row."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QPushButton,
)


class CalibrationPanel(QWidget):
    calibration_changed = Signal(float, float)

    def __init__(self, parent=None, pitch_deg=-7.0, height_m=1.0):
        super().__init__(parent)
        self.pitch_deg = float(pitch_deg)
        self.height_m = float(height_m)
        self._default_pitch = self.pitch_deg
        self._default_height = self.height_m
        self.setMaximumHeight(36)
        self.init_ui()

    def init_ui(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        self.setStyleSheet("""
            QLabel { color: #8B949E; font-size: 10px; }
            QLabel#val { color: #58A6FF; font-weight: bold; font-size: 10px; min-width: 42px; }
            QSlider::groove:horizontal {
                height: 3px; background: #2D3440; border-radius: 1px;
            }
            QSlider::handle:horizontal {
                background: #58A6FF; width: 10px; height: 10px;
                margin: -4px 0; border-radius: 5px;
            }
            QPushButton {
                background-color: #21262D; color: #C9D1D9;
                border: 1px solid #30363D; border-radius: 3px;
                padding: 2px 8px; font-size: 10px;
            }
            QPushButton:hover { border-color: #58A6FF; }
        """)

        row.addWidget(QLabel("Pitch"))
        self.slider_pitch = QSlider(Qt.Horizontal)
        self.slider_pitch.setRange(-120, 20)
        self.slider_pitch.setValue(int(self.pitch_deg * 10))
        self.slider_pitch.setMaximumWidth(140)
        self.lbl_pitch = QLabel(f"{self.pitch_deg:.1f}°")
        self.lbl_pitch.setObjectName("val")
        self.slider_pitch.valueChanged.connect(self.on_slider_changed)
        row.addWidget(self.slider_pitch)
        row.addWidget(self.lbl_pitch)

        row.addWidget(QLabel("H"))
        self.slider_height = QSlider(Qt.Horizontal)
        self.slider_height.setRange(10, 30)
        self.slider_height.setValue(int(self.height_m * 10))
        self.slider_height.setMaximumWidth(100)
        self.lbl_height = QLabel(f"{self.height_m:.1f}m")
        self.lbl_height.setObjectName("val")
        self.slider_height.valueChanged.connect(self.on_slider_changed)
        row.addWidget(self.slider_height)
        row.addWidget(self.lbl_height)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset to OpenLane extrinsics (−3° / 1.5 m). Live retune is locked.")
        btn_reset.clicked.connect(self.reset_defaults)
        row.addWidget(btn_reset)
        row.addStretch()

    def on_slider_changed(self, _=None):
        self.pitch_deg = self.slider_pitch.value() / 10.0
        self.height_m = self.slider_height.value() / 10.0
        self.lbl_pitch.setText(f"{self.pitch_deg:.1f}°")
        self.lbl_height.setText(f"{self.height_m:.1f}m")
        self.calibration_changed.emit(self.pitch_deg, self.height_m)

    def reset_defaults(self):
        self.slider_pitch.setValue(int(self._default_pitch * 10))
        self.slider_height.setValue(int(self._default_height * 10))
