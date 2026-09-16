#!/usr/bin/env python3
"""
YOLO localizes US speed signs; NVIDIA LPRNet reads the digits.

Runs on GRMN6693 around the known SPEED LIMIT 55 plate, then reports
string / mph vs YOLO class, plus TensorRT latency.

Example:
  python scripts/test_lprnet.py
  python scripts/test_lprnet.py --video testing_new_videos/GRMN6693_540_nohud.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.lprnet import CROP_MODES
from src.inference.speed_limit_tracker import class_to_mph
from src.inference.traffic_sign_detector import (
    TrafficSignDetector,
    filter_sign_dets,
    letterbox_stack,
)


def _release_yolo(detector):
    try:
        del detector.model
    except Exception:
        pass
    del detector
    try:
        import torch
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def collect_sign_crops(video, every, conf, start, end, max_dets):
    detector = TrafficSignDetector(conf_thresh=conf)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows = []
    infer_ms = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi < start:
            fi += 1
            continue
        if end > 0 and fi > end:
            break
        if fi % every != 0:
            fi += 1
            continue
        t0 = time.perf_counter()
        dets, _ = detector.detect(frame)
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        dets = filter_sign_dets(dets, frame.shape)
        speed = [d for d in dets if class_to_mph(d.get("class", "")) is not None]
        use = speed or dets
        for det in use:
            rows.append(
                {
                    "frame": fi,
                    "bbox": [int(x) for x in det["bbox"]],
                    "yolo_class": det["class"],
                    "yolo_conf": float(det["conf"]),
                    "yolo_mph": class_to_mph(det.get("class", "")),
                    "frame_bgr": frame.copy(),
                }
            )
            if max_dets and len(rows) >= max_dets:
                cap.release()
                yolo_mean = float(np.mean(infer_ms)) if infer_ms else 0.0
                _release_yolo(detector)
                return rows, yolo_mean, fps, n_frames
        fi += 1
        if max_dets and len(rows) >= max_dets:
            break
    cap.release()
    yolo_mean = float(np.mean(infer_ms)) if infer_ms else 0.0
    _release_yolo(detector)
    return rows, yolo_mean, fps, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="testing_new_videos/GRMN6693_540_nohud.mp4")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--start", type=int, default=880)
    ap.add_argument("--end", type=int, default=1020)
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--outdir", default="output/lprnet")
    args = ap.parse_args()

    video = args.video if os.path.isabs(args.video) else os.path.join(ROOT, args.video)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"[yolo] {video} frames {args.start}-{args.end} every {args.every}", flush=True)
    rows, yolo_ms, fps, n_frames = collect_sign_crops(
        video, args.every, args.conf, args.start, args.end, args.max_dets
    )
    print(f"[yolo] {len(rows)} speed-sign crops  infer mean {yolo_ms:.1f} ms", flush=True)

    from src.inference.lprnet import LPRNetRecognizer

    rec = LPRNetRecognizer(engine_path=args.engine)
    print(f"[lpr] engine={rec.engine_path}", flush=True)

    results = []
    crops_for_sheet = []
    bench_crop = None
    hits_55 = {m: 0 for m in CROP_MODES}

    for row in rows:
        frame = row.pop("frame_bgr")
        recs = {}
        for mode in CROP_MODES:
            out = rec.read_bbox(frame, row["bbox"], mode=mode)
            recs[mode] = {
                "text": out["text"],
                "conf": round(out["conf"], 3),
                "mph": out["mph"],
            }
            if out["mph"] == 55:
                hits_55[mode] += 1
            if mode == "full":
                crop = out["crop"]
                vis = crop.copy()
                label = f"{out['text'] or '?'} {out['mph'] or '--'}"
                cv2.putText(
                    vis, label, (2, max(12, vis.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1, cv2.LINE_AA,
                )
                crops_for_sheet.append(vis)
                if bench_crop is None and out["mph"] == 55:
                    bench_crop = crop
                elif bench_crop is None:
                    bench_crop = crop
                crop_path = os.path.join(outdir, f"f{row['frame']:04d}_{mode}.jpg")
                cv2.imwrite(crop_path, crop)
        item = {
            "frame": row["frame"],
            "bbox": row["bbox"],
            "yolo_class": row["yolo_class"],
            "yolo_conf": round(row["yolo_conf"], 3),
            "yolo_mph": row["yolo_mph"],
            "lpr": recs,
        }
        results.append(item)
        d = recs["full"]
        print(
            f"  f={row['frame']:4d}  yolo={row['yolo_class']:16s} {row['yolo_conf']:.2f}"
            f"  lpr[{d['text']!r}] mph={d['mph']} conf={d['conf']:.2f}"
            f"  digits={recs['digits']['text']!r}",
            flush=True,
        )

    if bench_crop is None:
        bench_crop = np.zeros((48, 96, 3), dtype=np.uint8)
        print("[lpr] no YOLO crops — latency on empty tensor", flush=True)

    print(f"[bench] warmup={args.warmup} iters={args.iters}", flush=True)
    bench = rec.benchmark(bench_crop, warmup=args.warmup, iters=args.iters)

    n = max(1, len(results))
    full_ok = sum(1 for r in results if r["lpr"]["full"]["mph"] is not None)
    full_55 = sum(1 for r in results if r["lpr"]["full"]["mph"] == 55)
    digits_ok = sum(1 for r in results if r["lpr"]["digits"]["mph"] is not None)
    digits_55 = sum(1 for r in results if r["lpr"]["digits"]["mph"] == 55)
    yolo_55 = sum(1 for r in results if r["yolo_mph"] == 55)
    disagree = sum(
        1
        for r in results
        if r["lpr"]["full"]["mph"] is not None and r["yolo_mph"] != r["lpr"]["full"]["mph"]
    )

    report = {
        "video": video,
        "engine": rec.engine_path,
        "yolo_infer_ms": round(yolo_ms, 2),
        "n_crops": len(results),
        "yolo_said_55": yolo_55,
        "lpr_full_parsed": full_ok,
        "lpr_full_said_55": full_55,
        "lpr_digits_parsed": digits_ok,
        "lpr_digits_said_55": digits_55,
        "yolo_lpr_disagree": disagree,
        "hits_55_by_crop": hits_55,
        "latency": {
            "infer_mean_ms": round(bench["infer"]["mean_ms"], 3),
            "infer_p50_ms": round(bench["infer"]["p50_ms"], 3),
            "infer_p95_ms": round(bench["infer"]["p95_ms"], 3),
            "wall_mean_ms": round(bench["preprocess_plus_infer"]["mean_ms"], 3),
            "qps": round(bench["qps"], 1),
            "warmup": bench["warmup"],
            "iters": bench["iters"],
        },
        "reads": results,
    }

    json_path = os.path.join(outdir, "GRMN6693_lprnet.json")
    txt_path = os.path.join(outdir, "GRMN6693_lprnet.txt")
    sheet_path = os.path.join(outdir, "GRMN6693_lprnet_sheet.jpg")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if crops_for_sheet:
        cv2.imwrite(sheet_path, letterbox_stack(crops_for_sheet, cols=6, cell=160, pad=6))

    lines = [
        "GRMN6693 LPRNet digit read (YOLO box → CTC)",
        f"video      : {video}",
        f"engine     : {rec.engine_path}",
        f"crops      : {len(results)}  YOLO infer {yolo_ms:.1f} ms",
        f"YOLO 55    : {yolo_55}/{n}",
        f"LPR full   : parsed {full_ok}/{n}  said 55 {full_55}/{n}  vs YOLO disagree {disagree}",
        f"LPR digits : parsed {digits_ok}/{n}  said 55 {digits_55}/{n}",
        f"hits 55    : {hits_55}",
        "",
        "Latency (TensorRT FP16, batch 1, 48x96)",
        f"  infer      mean {bench['infer']['mean_ms']:.2f} ms  "
        f"p50 {bench['infer']['p50_ms']:.2f}  p95 {bench['infer']['p95_ms']:.2f}",
        f"  +preprocess mean {bench['preprocess_plus_infer']['mean_ms']:.2f} ms",
        f"  throughput {bench['qps']:.0f} qps",
        "",
        "Per-crop (full sign, letterbox 48x96)",
        "-" * 72,
    ]
    for r in results:
        d = r["lpr"]["full"]
        lines.append(
            f"  f={r['frame']:4d}  yolo={r['yolo_class']:16s} {r['yolo_conf']:.2f}"
            f"  lpr={d['text']!r:8s} mph={str(d['mph']):>4s} conf={d['conf']:.2f}"
            f"  digits={r['lpr']['digits']['text']!r}"
        )
    text = "\n".join(lines) + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"[wrote] {txt_path}")
    print(f"[wrote] {json_path}")
    if crops_for_sheet:
        print(f"[wrote] {sheet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
