#!/usr/bin/env python3
"""Offscreen sweep of Tesla RuntimeLoader Euler rest poses."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("QSG_RHI_BACKEND", "opengl")

from PySide6.QtCore import QTimer
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtWidgets import QApplication

from src.ui.bev_quick3d import BevQuick3DWidget

POSES = [
    (-90, -90, 0),
    (-90, 90, 0),
    (-90, 180, 0),
    (-90, 0, 0),
    (0, 90, -90),
    (0, -90, -90),
]


def main():
    QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    app = QApplication(sys.argv)
    w = BevQuick3DWidget()
    w.resize(640, 720)
    w.show()
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "tesla_sweep"))
    os.makedirs(out_dir, exist_ok=True)

    w.update_bev_data([], [
        {"X_3d": -4.0, "Z_3d": 16.0, "track_id": 1, "label": "car"},
    ], "SAFE")

    state = {"i": -1, "wait": 0}

    def tick():
        root = w.rootObject()
        if root is None:
            return
        if state["wait"] > 0:
            state["wait"] -= 1
            if state["wait"] == 0:
                i = state["i"]
                rx, ry, rz = POSES[i]
                img = w.grabFramebuffer()
                if img.isNull():
                    img = w.grab().toImage()
                path = os.path.join(out_dir, f"rx{rx}_ry{ry}_rz{rz}.png")
                img.save(path)
                print("saved", path, img.width(), img.height())
            return
        state["i"] += 1
        if state["i"] >= len(POSES):
            app.quit()
            return
        rx, ry, rz = POSES[state["i"]]
        root.setProperty("teslaScale", 0.008)
        root.setProperty("teslaRotX", float(rx))
        root.setProperty("teslaRotY", float(ry))
        root.setProperty("teslaRotZ", float(rz))
        root.setProperty("overlayHint", f"Tesla rest rx{rx} ry{ry} rz{rz}")
        print("pose", rx, ry, rz)
        state["wait"] = 8

    timer = QTimer()
    timer.setInterval(80)
    timer.timeout.connect(tick)
    timer.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
