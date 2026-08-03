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
from src.ui.main_window import ADASMainWindow

def main():
    parser = argparse.ArgumentParser(description="PySide6 ADAS & 3D BEV Visualization Dashboard")
    parser.add_argument("--video", type=str, default="data/images/example_3.mp4", help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/anchor3dlane_raw.engine", help="Path to TensorRT engine")

    parser.add_argument("--test-mode", action="store_true", help="Run automated headless test mode and exit cleanly")
    args = parser.parse_args()

    if args.test_mode:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication(sys.argv)
    window = ADASMainWindow(video_path=args.video, model_path=args.model)

    if args.test_mode:
        print("[PySide6 App Test] Initializing GUI in offscreen mode...")
        # Auto-shutdown after 3 seconds in test mode
        QTimer.singleShot(3000, lambda: (print("[PySide6 App Test] Test Passed Cleanly!"), app.quit()))

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
