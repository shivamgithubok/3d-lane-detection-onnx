"""
PySide6 Main ADAS Cockpit Window
Split-screen Front Camera + Dynamic BEV Canvas + Extrinsics Control Panel.
"""

import sys
import numpy as np
from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QStatusBar, QFrame, QFileDialog
)
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon

from src.ui.worker import InferenceWorker
from src.ui.bev_quick3d import BevQuick3DWidget, create_bev_widget
from src.ui.calibration_panel import CalibrationPanel
from src.utils.calibration import preset_for_video

class ADASMainWindow(QMainWindow):
    def __init__(self, video_path=None, model_path="models/anchor3dlane_raw.engine", bev_backend="quick3d"):

        super().__init__()
        self.setWindowTitle("Futuristic 3D Lane & ADAS Cockpit (PySide6)")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        self.video_path = video_path
        self.model_path = model_path
        self.bev_backend = bev_backend
        self.preset_pitch, self.preset_height = preset_for_video(video_path)

        # Setup Theme & Layout
        self.apply_dark_theme()
        self.init_ui()

        # Initialize Async Inference Worker Thread
        self.worker = InferenceWorker(video_path=self.video_path, model_path=self.model_path)
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.status_message.connect(self.on_status_message)
        # Cal shows OpenLane defaults; P is locked in the worker (do not retune for Garmin).
        # Do NOT push Cal pitch into BEV calibPitch — that is a view tilt, not extrinsics.
        if self.calib_panel is not None:
            self.calib_panel.calibration_changed.connect(self.on_calibration_changed)
            self.worker.set_calibration(self.calib_panel.pitch_deg, self.calib_panel.height_m)
        self.worker.start()

    def on_calibration_changed(self, pitch_deg, height_m):
        # Snap UI back to training extrinsics; never rebuild P from free sliders.
        if abs(pitch_deg - self.preset_pitch) > 1e-3 or abs(height_m - self.preset_height) > 1e-3:
            self.calib_panel.blockSignals(True)
            self.calib_panel.reset_defaults()
            self.calib_panel.blockSignals(False)
            self.statusBar().showMessage(
                "P locked to OpenLane (−3° / 1.5 m) — retuning breaks corridor projection",
                4000,
            )
        self.worker.set_calibration(self.preset_pitch, self.preset_height)

    def apply_dark_theme(self):
        """Applies a sleek, dark ADAS futuristic theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0D1117;
            }
            QWidget {
                color: #C9D1D9;
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }
            QFrame#header_bar {
                background-color: #161B22;
                border-bottom: 1px solid #30363D;
            }
            QLabel#title_label {
                color: #58A6FF;
                font-weight: bold;
                font-size: 14px;
            }
            QLabel#hud_badge {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton#ctrl_btn {
                background-color: #21262D;
                color: #C9D1D9;
                border: 1px solid #30363D;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton#ctrl_btn:hover {
                background-color: #30363D;
                border-color: #58A6FF;
                color: #FFFFFF;
            }
            QStatusBar {
                background-color: #161B22;
                color: #8B949E;
                border-top: 1px solid #30363D;
            }
        """)

    def init_ui(self):
        main_central = QWidget()
        self.setCentralWidget(main_central)
        main_layout = QVBoxLayout(main_central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Header Navigation Bar
        header = QFrame()
        header.setObjectName("header_bar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 4, 12, 4)

        lbl_title = QLabel("🛣️ 3D LANE DETECTION & BEV ADAS VISUALIZER")
        lbl_title.setObjectName("title_label")
        header_layout.addWidget(lbl_title)

        header_layout.addStretch()

        # Telemetry HUD Badges
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setObjectName("hud_badge")
        self.lbl_latency = QLabel("Latency: -- ms")
        self.lbl_latency.setObjectName("hud_badge")
        self.lbl_status_hud = QLabel("Status: INITIALIZING")
        self.lbl_status_hud.setObjectName("hud_badge")

        header_layout.addWidget(self.lbl_fps)
        header_layout.addWidget(self.lbl_latency)
        header_layout.addWidget(self.lbl_status_hud)

        main_layout.addWidget(header)

        # 2. Main Viewports Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #30363D; }")

        # Left Container: Front Camera Feed & Video Controls
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(10, 10, 5, 10)

        # Camera Frame Label
        self.lbl_camera = QLabel("Camera Feed Loading...")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #000000; border: 1px solid #21262D; border-radius: 6px;")
        self.lbl_camera.setMinimumSize(480, 360)
        self.lbl_camera.setScaledContents(True)
        left_layout.addWidget(self.lbl_camera, stretch=1)


        # Video Control Buttons
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("⏸ Pause")
        self.btn_play.setObjectName("ctrl_btn")
        self.btn_play.clicked.connect(self.toggle_play_pause)

        self.btn_open_video = QPushButton("📁 Open Video")
        self.btn_open_video.setObjectName("ctrl_btn")
        self.btn_open_video.clicked.connect(self.open_video_file)

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_open_video)
        ctrl_layout.addStretch()
        left_layout.addLayout(ctrl_layout)

        splitter.addWidget(left_container)

        # Right Container: BEV Canvas & Extrinsics Calibration Panel
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 8, 4)
        right_layout.setSpacing(0)

        # Interactive BEV Widget (Qt Quick 3D or QPainter fallback)
        self.bev_widget = create_bev_widget(self.bev_backend)
        right_layout.addWidget(self.bev_widget, stretch=1)

        self.btn_road_style = None
        self.btn_lane_lines = None
        # Always show extrinsics (P2) — drives front P_matrix + BEV camera
        bev_ctrl_layout = QHBoxLayout()
        if not isinstance(self.bev_widget, BevQuick3DWidget):
            btn_reset_bev = QPushButton("Reset BEV")
            btn_reset_bev.setObjectName("ctrl_btn")
            btn_reset_bev.clicked.connect(self.bev_widget.reset_view)
            bev_ctrl_layout.addWidget(btn_reset_bev)
            self.btn_road_style = QPushButton("Road: Cinematic")
            self.btn_road_style.setObjectName("ctrl_btn")
            self.btn_road_style.clicked.connect(self.toggle_road_style)
            bev_ctrl_layout.addWidget(self.btn_road_style)
            self.btn_lane_lines = QPushButton("Lanes: ON")
            self.btn_lane_lines.setObjectName("ctrl_btn")
            self.btn_lane_lines.setCheckable(True)
            self.btn_lane_lines.setChecked(True)
            self.btn_lane_lines.clicked.connect(self.toggle_lane_lines)
            bev_ctrl_layout.addWidget(self.btn_lane_lines)
        bev_ctrl_layout.addStretch()
        self.calib_panel = CalibrationPanel(
            pitch_deg=self.preset_pitch, height_m=self.preset_height
        )
        bev_ctrl_layout.addWidget(self.calib_panel)
        right_layout.addLayout(bev_ctrl_layout)

        splitter.addWidget(right_container)
        splitter.setSizes([640, 640])

        main_layout.addWidget(splitter, stretch=1)

        # 3. Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. PySide6 Engine active.")

    @Slot(np.ndarray, list, list, object, str, object, object, float, float)
    def on_frame_processed(self, frame_rgb, proposals, processed_objs, cipo_obj, cipo_status, left_3d, right_3d, fps, latency_ms):
        """Callback invoked when worker thread emits a newly processed frame."""
        # 1. Update Camera Video Label
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_camera.setPixmap(pixmap)


        # 2. Update BEV Canvas Widget
        self.bev_widget.update_bev_data(proposals, processed_objs, cipo_status, left_3d, right_3d)

        # 3. Update HUD Badges
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_latency.setText(f"Latency: {latency_ms:.1f} ms")

        if cipo_obj:
            cipo_str = f"{cipo_obj['Z_3d']:.1f}m [{cipo_status}]"
        else:
            cipo_str = f"{cipo_status}"

        self.lbl_status_hud.setText(f"CIPO: {cipo_str}")

        if cipo_status == "DANGER":
            self.lbl_status_hud.setStyleSheet("background-color: #DA3633; color: #FFFFFF;")
        elif cipo_status == "WARNING":
            self.lbl_status_hud.setStyleSheet("background-color: #D9822B; color: #FFFFFF;")
        elif cipo_status == "DEGRADED":
            self.lbl_status_hud.setStyleSheet("background-color: #6E7681; color: #FFFFFF;")
        else:
            self.lbl_status_hud.setStyleSheet("background-color: #238636; color: #FFFFFF;")


    @Slot(str)
    def on_status_message(self, msg):
        self.status_bar.showMessage(msg)

    def toggle_play_pause(self):
        is_paused = self.worker.toggle_pause()
        if is_paused:
            self.btn_play.setText("▶ Play")
        else:
            self.btn_play.setText("⏸ Pause")

    def toggle_road_style(self):
        cinematic = self.bev_widget.toggle_cinematic_road()
        if self.btn_road_style is not None:
            self.btn_road_style.setText("Road: Cinematic" if cinematic else "Road: Grid")

    def toggle_lane_lines(self):
        show = self.bev_widget.toggle_lane_lines()
        if self.btn_lane_lines is not None:
            self.btn_lane_lines.setChecked(show)
            self.btn_lane_lines.setText("Lanes: ON" if show else "Lanes: OFF")

    def open_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open MP4 Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.worker.stop()
            self.worker = InferenceWorker(video_path=file_name, model_path=self.model_path)
            self.worker.frame_processed.connect(self.on_frame_processed)
            self.worker.status_message.connect(self.on_status_message)
            self.worker.start()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()
