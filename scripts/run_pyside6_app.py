#!/usr/bin/env python3
"""
PySide6 ADAS Visualization Dashboard Executable Launcher
Usage:
    python scripts/run_pyside6_app.py --video data/images/example_3.mp4
    python scripts/run_pyside6_app.py --test-mode
"""

import sys
import os
import argparse

# Add project root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from src.ui.main_window import ADASMainWindow


def _configure_quick3d_graphics():
    """Prefer OpenGL on Jetson; must run before QApplication."""
    os.environ.setdefault("QSG_RHI_BACKEND", "opengl")
    try:
        QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="PySide6 ADAS & 3D BEV Visualization Dashboard")
    parser.add_argument("--video", type=str, default="data/images/example_3.mp4", help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/anchor3dlane_raw.engine", help="Path to TensorRT engine")
    parser.add_argument(
        "--bev",
        choices=("quick3d", "painter"),
        default="quick3d",
        help="BEV viewport: Qt Quick 3D (default) or legacy QPainter sprites",
    )
    parser.add_argument("--test-mode", action="store_true", help="Run automated headless test mode and exit cleanly")
    args = parser.parse_args()

    if args.test_mode:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        # Offscreen + Quick 3D is unreliable; default the smoke test to painter
        # unless the caller explicitly asked for quick3d.
        if args.bev == "quick3d" and "--bev" not in sys.argv:
            args.bev = "painter"

    _configure_quick3d_graphics()
    app = QApplication(sys.argv)
    window = ADASMainWindow(video_path=args.video, model_path=args.model, bev_backend=args.bev)

    if args.test_mode:
        print("[PySide6 App Test] Initializing GUI in offscreen mode...")
        # Auto-shutdown after 3 seconds in test mode
        QTimer.singleShot(3000, lambda: (print("[PySide6 App Test] Test Passed Cleanly!"), app.quit()))

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
