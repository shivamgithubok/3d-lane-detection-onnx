"""
PySide6 Main ADAS Cockpit Window
Hexagonal infotainment cluster: BEV / live camera + HUD chrome.
"""

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QStatusBar
)
from PySide6.QtGui import QImage, QPixmap

from src.ui.worker import InferenceWorker
from src.ui.bev_quick3d import BevQuick3DWidget, create_bev_widget
from src.ui.calibration_panel import CalibrationPanel
from src.ui.hex_cockpit import WINDOW_H, WINDOW_W, HexCockpit
from src.utils.calibration import preset_for_video

class ADASMainWindow(QMainWindow):
    def __init__(self, video_path=None, model_path="models/anchor3dlane_raw.engine", bev_backend="quick3d"):

        super().__init__()
        self.setWindowTitle("ADAS Infotainment Cluster")
        self.setFixedSize(WINDOW_W, WINDOW_H)

        self.video_path = video_path
        self.model_path = model_path
        self.bev_backend = bev_backend
        self.preset_pitch, self.preset_height = preset_for_video(video_path)

        self.apply_dark_theme()
        self.init_ui()

        # Initialize Async Inference Worker Thread
        self.worker = InferenceWorker(video_path=self.video_path, model_path=self.model_path)
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.status_message.connect(self.on_status_message)
        # Cal sliders retune object GroundCalibration. P / BEV view stay untouched.
        if self.calib_panel is not None:
            self.calib_panel.set_from_calib(self.worker.ground_calib)
            self.calib_panel.calibration_changed.connect(self.on_calibration_changed)
        self.worker.start()

    def on_calibration_changed(self, pitch_deg, height_m):
        self.worker.set_object_calib(pitch_deg, height_m)
        self.statusBar().showMessage(
            f"Object range: pitch {pitch_deg:.1f}°  h {height_m:.2f} m",
            2000,
        )

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #050A14; }
            QWidget { color: #C9D1D9; font-family: 'Inter', 'Segoe UI', sans-serif; }
            QPushButton#ctrl_btn {
                background-color: #101A28; color: #C8E8FF;
                border: 1px solid #2A4A62; border-radius: 3px;
                padding: 2px 8px; font-size: 11px;
            }
            QPushButton#ctrl_btn:hover { border-color: #3EC8FF; }
            QStatusBar {
                background-color: #050A14; color: #7A93A8;
                border-top: 1px solid #1A3048;
            }
        """)

    def init_ui(self):
        main_central = QWidget()
        self.setCentralWidget(main_central)
        main_layout = QVBoxLayout(main_central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.bev_widget = create_bev_widget(self.bev_backend)
        if isinstance(self.bev_widget, BevQuick3DWidget):
            self.bev_widget.enable_cluster_chrome()

        self.cockpit = HexCockpit(self.bev_widget)
        self.lbl_camera = self.cockpit.lbl_camera
        self.btn_play = self.cockpit.btn_play
        self.btn_open_video = self.cockpit.btn_open
        self.btn_play.clicked.connect(self.toggle_play_pause)
        self.btn_open_video.clicked.connect(self.open_video_file)

        self.calib_panel = CalibrationPanel(
            pitch_deg=self.preset_pitch, height_m=self.preset_height
        )
        self.cockpit.mode_panel.layout().addWidget(self.calib_panel)

        main_layout.addWidget(self.cockpit, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.hide()

    @Slot(np.ndarray, list, list, object, str, object, object, float, float, object, float, object)
    def on_frame_processed(self, frame_rgb, proposals, processed_objs, cipo_obj, cipo_status,
                           left_3d, right_3d, fps, latency_ms, speed_mps=None, source_dt=1.0 / 30.0,
                           alerts=None):
        if self.cockpit._view == "live":
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.lbl_camera.setPixmap(QPixmap.fromImage(q_img))

        if self.cockpit._view == "bev":
            self.bev_widget.update_bev_data(
                proposals, processed_objs, cipo_status, left_3d, right_3d,
                speed_mps=speed_mps, dt=source_dt, alerts=alerts,
            )
        self.cockpit.update_hud(
            speed_mps=speed_mps,
            cipo_obj=cipo_obj,
            cipo_status=cipo_status,
            alerts=alerts,
            fps=fps,
            lane_ok=(left_3d is not None and right_3d is not None),
        )

    @Slot(str)
    def on_status_message(self, msg):
        if self.status_bar.isVisible():
            self.status_bar.showMessage(msg)

    def toggle_play_pause(self):
        is_paused = self.worker.toggle_pause()
        self.btn_play.setText("▶" if is_paused else "⏸")

    def open_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open MP4 Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.worker.stop()
            self.worker = InferenceWorker(video_path=file_name, model_path=self.model_path)
            self.worker.frame_processed.connect(self.on_frame_processed)
            self.worker.status_message.connect(self.on_status_message)
            if self.calib_panel is not None:
                self.calib_panel.set_from_calib(self.worker.ground_calib)
            self.worker.start()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()
