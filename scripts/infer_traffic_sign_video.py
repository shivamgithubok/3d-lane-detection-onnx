#!/usr/bin/env python3
"""
OCR Garmin HUD speed, crop the HUD, run scheduled traffic-sign TensorRT.

Example:
  python scripts/infer_traffic_sign_video.py testing_new_videos/GRMN6693_540.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.speed_limit_tracker import SpeedLimitTracker, class_to_mph
from src.inference.traffic_sign_detector import (
    TrafficSignDetector,
    draw_isa_overlay,
    filter_sign_dets,
    resolve_model_path,
)
from src.utils.ego_speed import (
    EgoSpeedLog,
    default_speed_json_path,
    export_nohud_video,
    extract_speed_log,
    save_speed_log,
)


def _write_report(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "GRMN traffic-sign video test",
        f"source     : {stats['source']}",
        f"nohud      : {stats['nohud']}",
        f"speed json : {stats['speed_json']}",
        f"model      : {stats['model']}",
        f"frames     : {stats['frames']}  sign_infer={stats['sign_infer_frames']}",
        f"OCR valid  : {stats['ocr_valid']} / {stats['ocr_frames']}  "
        f"mph {stats['ocr_min']}–{stats['ocr_max']}",
        f"sign every : {stats['every']}  infer mean {stats['infer_ms']:.1f} ms",
        f"end-to-end : {stats['e2e_ms']:.1f} ms/frame  ({stats['fps']:.1f} FPS write)",
        "",
        f"Confirmed limits: {stats['confirmed'] or 'none'}",
        f"Classes seen    : {', '.join(stats['classes']) or 'none'}",
        "",
        "Per-class raw counts (filtered boxes)",
        "-" * 48,
    ]
    for name, n in sorted(stats["class_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {name:22s} {n}")
    lines += ["", "Confirmed timeline (frame, mph, conf)", "-" * 48]
    for ev in stats["events"]:
        lines.append(f"  f={ev['frame']:4d}  {ev['mph']} mph  conf={ev['conf']:.2f}  {ev['status']}")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="testing_new_videos/GRMN6693_540.mp4")
    ap.add_argument("--model", default=None)
    ap.add_argument("--every", type=int, default=2, help="run sign model every N frames")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--act-conf", type=float, default=0.55)
    ap.add_argument("--hood-frac", type=float, default=0.08)
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    os.chdir(ROOT)
    src = os.path.abspath(args.video)
    if not os.path.isfile(src):
        print(f"Error: video not found: {src}")
        return 1

    speed_json = default_speed_json_path(src)
    if args.skip_ocr and os.path.isfile(speed_json):
        print(f"[ocr] using {speed_json}")
    else:
        print(f"[ocr] reading HUD speed from {src}")
        log = extract_speed_log(src, max_frames=args.max_frames)
        save_speed_log(log, speed_json)
        print(
            f"[ocr] wrote {speed_json}  valid={log['valid_frames']}/{log['frame_count']}"
        )

    speed_log = EgoSpeedLog.from_json(speed_json)
    ocr_mph = []
    if os.path.isfile(speed_json):
        with open(speed_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ocr_mph = [v for v in raw.get("mph") or [] if v is not None]

    stem, ext = os.path.splitext(src)
    nohud = stem + "_nohud" + (ext or ".mp4")
    if args.skip_crop and os.path.isfile(nohud):
        print(f"[crop] using {nohud}")
        hud_y = None
        n_crop = 0
        wh = None
    else:
        print(f"[crop] HUD → {nohud}")
        n_crop, wh, hud_y = export_nohud_video(src, nohud, hood_frac=args.hood_frac)
        print(f"[crop] {n_crop} frames {wh} hud_y={hud_y}")

    model_path = args.model or resolve_model_path()
    print(f"[sign] {model_path}")
    detector = TrafficSignDetector(model_path=model_path, conf_thresh=args.conf)
    isa = SpeedLimitTracker(confirm_hits=3, act_conf=args.act_conf)

    cap = cv2.VideoCapture(nohud)
    if not cap.isOpened():
        print(f"Error: cannot open {nohud}")
        return 1
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = os.path.join(ROOT, "output", "traffic_sign")
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, "GRMN6693_540_signs.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    class_counts = {}
    classes_seen = set()
    events = []
    last_posted = None
    last_dets = []
    infer_ms = []
    t0 = time.perf_counter()
    i = 0
    sign_n = 0
    snap_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and i >= args.max_frames:
            break

        ran = (i % max(1, args.every)) == 0
        if ran:
            dets, speed = detector.detect(frame)
            dets = filter_sign_dets(dets, frame.shape)
            last_dets = dets
            infer_ms.append(float((speed or {}).get("inference", 0.0)))
            sign_n += 1
            for d in dets:
                class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
                classes_seen.add(d["class"])
            snap = isa.update(dets, ran_infer=True, min_conf=args.conf)
        else:
            snap = isa.update([], ran_infer=False)
            dets = last_dets

        if snap["posted_mph"] != last_posted and snap["status"] == "CONFIRMED":
            events.append(
                {
                    "frame": i,
                    "mph": snap["posted_mph"],
                    "conf": round(float(snap["conf"]), 3),
                    "status": snap["status"],
                }
            )
            last_posted = snap["posted_mph"]

        ego = speed_log.get_mph(i)
        vis = draw_isa_overlay(frame, snap, ego_mph=ego, detections=dets)
        writer.write(vis)
        if ran and dets and snap_n < 12:
            cv2.imwrite(os.path.join(out_dir, f"grmn6693_f{i:04d}.jpg"), vis)
            snap_n += 1
        i += 1
        if i % 200 == 0:
            print(f"[sign] frame {i} posted={snap['posted_mph']} status={snap['status']}")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    confirmed = sorted({e["mph"] for e in events if e["mph"] is not None})
    stats = {
        "source": src,
        "nohud": nohud,
        "speed_json": speed_json,
        "model": str(detector.model_path),
        "frames": i,
        "sign_infer_frames": sign_n,
        "ocr_frames": len(speed_log.mps),
        "ocr_valid": len(ocr_mph),
        "ocr_min": min(ocr_mph) if ocr_mph else None,
        "ocr_max": max(ocr_mph) if ocr_mph else None,
        "every": args.every,
        "infer_ms": sum(infer_ms) / max(1, len(infer_ms)),
        "e2e_ms": (elapsed / max(1, i)) * 1000.0,
        "fps": i / max(elapsed, 1e-6),
        "confirmed": confirmed,
        "classes": sorted(classes_seen),
        "class_counts": class_counts,
        "events": events,
        "hud_y": hud_y,
        "crop_wh": wh,
        "crop_frames": n_crop,
    }
    report_path = os.path.join(out_dir, "GRMN6693_540_report.txt")
    print(_write_report(report_path, stats))
    with open(os.path.join(out_dir, "GRMN6693_540_report.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] {out_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
