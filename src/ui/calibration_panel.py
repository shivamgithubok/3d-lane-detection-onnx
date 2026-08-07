"""
PySide6 Live Extrinsics Calibration Panel
Interactive sliders for real-time Pitch, Roll, Yaw, Camera Height tuning.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox
)

class CalibrationPanel(QWidget):
    # Signal emitted when parameters change
    calibration_changed = Signal(float, float) # pitch_deg, height_m

    def __init__(self, parent=None):
        super().__init__(parent)
        # Tuned defaults matching the preferred BEV look
        self.pitch_deg = -7.0
        self.height_m = 1.0
        self.roll_deg = 0.0
        self.yaw_deg = 0.0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        group_box = QGroupBox("🛠️ Camera Extrinsics & Calibration Tuner")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00C8FF;
                border: 1px solid #2D3440;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #D0D7DE;
                font-size: 11px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2D3440;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00C8FF;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QPushButton {
                background-color: #21262D;
                color: #FFFFFF;
                border: 1px solid #30363D;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363D;
                border-color: #00C8FF;
            }
        """)

        grid_layout = QVBoxLayout(group_box)

        # 1. Pitch Angle Slider (-10 deg to +10 deg)
        self.lbl_pitch, self.slider_pitch = self.create_slider_row(
            "Pitch Angle (θ):", -100, 100, int(self.pitch_deg * 10), "°", self.on_slider_changed
        )
        grid_layout.addLayout(self.lbl_pitch)
        grid_layout.addWidget(self.slider_pitch)

        # 2. Camera Height Slider (1.0m to 3.0m)
        self.lbl_height, self.slider_height = self.create_slider_row(
            "Camera Height (H):", 10, 30, int(self.height_m * 10), "m", self.on_slider_changed
        )
        grid_layout.addLayout(self.lbl_height)
        grid_layout.addWidget(self.slider_height)

        # 3. Reset Button
        btn_reset = QPushButton("↺ Reset Extrinsics")
        btn_reset.clicked.connect(self.reset_defaults)
        grid_layout.addWidget(btn_reset)

        layout.addWidget(group_box)

    def create_slider_row(self, label_text, min_val, max_val, default_val, unit_str, callback):
        row = QHBoxLayout()
        lbl_title = QLabel(label_text)
        lbl_value = QLabel(f"{default_val/10.0:.1f}{unit_str}")
        lbl_value.setStyleSheet("color: #00C8FF; font-weight: bold;")
        row.addWidget(lbl_title)
        row.addStretch()
        row.addWidget(lbl_value)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(lambda v: self.update_slider_label(lbl_value, v / 10.0, unit_str, callback))

        return row, slider

    def update_slider_label(self, label, val, unit, callback):
        label.setText(f"{val:.1f}{unit}")
        callback()

    def on_slider_changed(self):
        self.pitch_deg = self.slider_pitch.value() / 10.0
        self.height_m = self.slider_height.value() / 10.0
        self.calibration_changed.emit(self.pitch_deg, self.height_m)

    def reset_defaults(self):
        self.slider_pitch.setValue(-70)   # -7.0°
        self.slider_height.setValue(10)   # 1.0 m
        self.on_slider_changed()
