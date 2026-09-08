#!/usr/bin/env python3
"""Render a few Qt Quick 3D frames and dump ego GLB fit + a screenshot."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("QSG_RHI_BACKEND", "opengl")

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtWidgets import QApplication

from src.ui.bev_quick3d import BevQuick3DWidget


def main():
    QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    app = QApplication(sys.argv)

    w = BevQuick3DWidget()
    w.resize(640, 720)
    w.setWindowTitle("ego GLB verify")
    w.show()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(out_dir, exist_ok=True)
    w.update_bev_data([], [
        {"X_3d": 0.0, "Z_3d": 14.0, "track_id": 1, "label": "car", "in_path": True, "show_bev": True, "lane_rank": 0},
        {"X_3d": -3.2, "Z_3d": 22.0, "track_id": 2, "label": "car", "in_path": False, "show_bev": True, "lane_rank": 1},
        {"X_3d": 3.1, "Z_3d": 30.0, "track_id": 3, "label": "truck", "in_path": False, "show_bev": True, "lane_rank": 1},
    ], "SAFE")
    out_png = os.path.join(out_dir, "verify_traffic_glb.png")
    frame_n = {"i": 0}

    def tick():
        root = w.rootObject()
        frame_n["i"] += 1
        i = frame_n["i"]
        debug = root.property("egoDebug") if root else None
        scale = root.property("egoScale") if root else None
        fitted = root.property("egoFitted") if root else None
        y = root.property("egoY") if root else None
        print(f"frame {i:02d}  fitted={fitted}  scale={scale}  y={y}  count={root.property('trafficCount')}  {debug}")
        if i < 12:
            return
        img = w.grabFramebuffer()
        if img.isNull():
            img = w.grab().toImage()
        img.save(out_png)
        print("saved", out_png, img.width(), img.height(), img.format())
        # Non-black check
        scaled = img.scaled(80, 90)
        bright = 0
        for yy in range(scaled.height()):
            for xx in range(scaled.width()):
                c = scaled.pixelColor(xx, yy)
                if c.red() + c.green() + c.blue() > 80:
                    bright += 1
        print(f"bright_pixels {bright}/{scaled.width()*scaled.height()}")
        app.quit()

    timer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(tick)
    timer.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
