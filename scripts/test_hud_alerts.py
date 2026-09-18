#!/usr/bin/env python3
"""HUD checks: posted limit is live mph, FCW/LDW only when triggered, layout grab."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QColor, QPainter
from src.ui.hex_cockpit import HexCockpit, WINDOW_W, WINDOW_H


class DummyBev(QWidget):
    def paintEvent(self, event):
        QPainter(self).fillRect(self.rect(), QColor("#071018"))


def main():
    app = QApplication(sys.argv)
    cockpit = HexCockpit(DummyBev())
    cockpit.resize(WINDOW_W, WINDOW_H)
    cockpit.show()

    cockpit.update_hud(speed_mps=13.4, alerts={"isa": {"posted_mph": 55}}, fps=12, lane_ok=True)
    html = cockpit.lbl_limit.text()
    assert "55" in html, html
    assert "89" not in html, html
    assert not cockpit.banner_fcw.isVisible()
    assert not cockpit.banner_ldw.isVisible()

    cockpit.update_hud(
        speed_mps=13.4,
        cipo_status="WARNING",
        cipo_obj={"Z_3d": 18.0},
        alerts={"isa": {"posted_mph": 55, "candidate_mph": 55}, "fcw": "FCW", "range_m": 18, "ldw": "LEFT", "ldw_side": "LEFT"},
        fps=12,
        lane_ok=True,
    )
    assert cockpit.banner_fcw.isVisible(), "FCW banner should show"
    assert cockpit.banner_ldw.isVisible(), "LDW banner should show"
    assert "55" in cockpit.lbl_limit.text()

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_hud_alerts_verify.png"))
    cockpit.grab().save(out)
    print("OK limit=55 FCW/LDW visible")
    print("WROTE", out)
    app.quit()


if __name__ == "__main__":
    main()
