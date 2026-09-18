#!/usr/bin/env python3
"""Offscreen grab of the hexagonal settings card stack."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter

from src.ui.hex_cockpit import HexCockpit, WINDOW_W, WINDOW_H


class DummyBev(QWidget):
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#071018"))
        p.setPen(QColor("#1a4a66"))
        for i in range(8):
            y = int(self.height() * (0.35 + i * 0.08))
            p.drawLine(0, y, self.width(), y)
        p.end()


def main():
    app = QApplication(sys.argv)
    dummy = DummyBev()
    cockpit = HexCockpit(dummy)
    cockpit.resize(WINDOW_W, WINDOW_H)
    cockpit.gauge.set_kmh(72)
    cockpit.lbl_limit.setText(
        "SPEED LIMIT<br><span style='font-size:26px;font-weight:800;color:#F4FAFF'>89</span><br>LIMIT"
    )
    cockpit.show()
    cockpit.settings_wheel.open_menu()
    cockpit.gear.set_active(True)
    out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tmp_settings_menu_verify.png")
    )

    def grab():
        pix = cockpit.grab()
        pix.save(out)
        print("WROTE", out)
        app.quit()

    QTimer.singleShot(450, grab)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
