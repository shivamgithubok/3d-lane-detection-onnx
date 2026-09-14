#!/usr/bin/env python3
"""
Standalone traffic-sign test (not part of the ADAS / CIPO loop).

Downloads the Hugging Face YOLOv11n weights, pulls public Wikimedia photos
covering the 12 LISA classes, runs detection, and reports latency.

Usage (from repo root, venv active):
    python scripts/test_traffic_sign.py
    python scripts/test_traffic_sign.py --export-engine
    python scripts/test_traffic_sign.py --model models/traffic_sign_yolo11n.engine
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.traffic_sign_detector import (
    TRAFFIC_SIGN_CLASSES,
    TrafficSignDetector,
    default_weight_paths,
    draw_detections,
    letterbox_stack,
    resolve_model_path,
)

USER_AGENT = "ElevaticsADAS-TSR/1.0 (standalone eval; research)"
IMAGE_DIR = os.path.join(ROOT, "data", "traffic_signs")
OUTPUT_DIR = os.path.join(ROOT, "output", "traffic_sign")

# Open-source photos: Openverse CC search + in-domain HF val sheets.
# expect=None means "any of the 13 classes" (used for model-card val mosaics).
TEST_CASES = [
    {
        "name": "hf_val_batch0.jpg",
        "expect": None,
        "hf_file": "val_batch0_pred.jpg",
    },
    {
        "name": "hf_val_batch1.jpg",
        "expect": None,
        "hf_file": "val_batch1_pred.jpg",
    },
    {
        "name": "hf_val_batch2.jpg",
        "expect": None,
        "hf_file": "val_batch2_pred.jpg",
    },
    {"name": "stop.jpg", "expect": "stop", "query": "united states octagonal stop sign road"},
    {"name": "yield.jpg", "expect": "yield", "query": "united states triangular yield sign road"},
    {"name": "do_not_enter.jpg", "expect": "doNotEnter", "query": "united states do not enter traffic sign"},
    {
        "name": "pedestrian_crossing.jpg",
        "expect": "pedestrianCrossing",
        "query": "united states pedestrian crossing yellow warning sign",
    },
    {"name": "speed_15.jpg", "expect": "speedLimit15", "query": "united states speed limit 15 mph sign"},
    {"name": "speed_25.jpg", "expect": "speedLimit25", "query": "united states speed limit 25 mph sign"},
    {"name": "speed_30.jpg", "expect": "speedLimit30", "query": "united states speed limit 30 mph sign"},
    {"name": "speed_35.jpg", "expect": "speedLimit35", "query": "united states speed limit 35 mph sign"},
    {"name": "speed_40.jpg", "expect": "speedLimit40", "query": "united states speed limit 40 mph sign"},
    {"name": "speed_45.jpg", "expect": "speedLimit45", "query": "united states speed limit 45 mph sign"},
    {"name": "speed_50.jpg", "expect": "speedLimit50", "query": "united states speed limit 50 mph sign"},
    {"name": "speed_55.jpg", "expect": "speedLimit55", "query": "united states speed limit 55 mph sign"},
    {"name": "speed_65.jpg", "expect": "speedLimit65", "query": "united states speed limit 65 mph highway sign"},
]


def _http_get(url: str, timeout=30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _save_image_bytes(dest: str, blob: bytes) -> bool:
    arr = np.frombuffer(blob, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        return False
    cv2.imwrite(dest, decoded)
    print(f"[img] wrote {dest} {decoded.shape[1]}x{decoded.shape[0]}")
    return True


def _download_hf_file(filename: str, dest: str) -> bool:
    from huggingface_hub import hf_hub_download

    from src.inference.traffic_sign_detector import HF_REPO

    cached = hf_hub_download(repo_id=HF_REPO, filename=filename)
    frame = cv2.imread(cached)
    if frame is None:
        return False
    cv2.imwrite(dest, frame)
    print(f"[img] {os.path.basename(dest)} ← HF {filename} {frame.shape[1]}x{frame.shape[0]}")
    return True


def _openverse_url(query: str) -> str | None:
    api = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(
        {
            "q": query,
            "page_size": 8,
            "filter_dead": "true",
        }
    )
    time.sleep(0.35)
    data = json.loads(_http_get(api, timeout=40).decode("utf-8"))
    for hit in data.get("results") or []:
        url = hit.get("url") or ""
        title = (hit.get("title") or "").lower()
        low = url.lower()
        if not url or low.endswith(".svg"):
            continue
        # Skip protest / unrelated "stop" hits.
        if any(bad in title for bad in ("tea party", "tar sands", "times square", "protest")):
            continue
        if any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
            return url
        if "flickr" in low or "staticflickr" in low or "upload.wikimedia" in low:
            return url
    return None


def split_val_mosaic(path: str, dest_dir: str, rows=4, cols=4, inset=4) -> list[str]:
    """Ultralytics val_batch_pred.jpg is a 4x4 contact sheet of LISA frames."""
    sheet = cv2.imread(path)
    if sheet is None:
        return []
    h, w = sheet.shape[:2]
    th, tw = h // rows, w // cols
    os.makedirs(dest_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * th + inset, c * tw + inset
            y2, x2 = (r + 1) * th - inset, (c + 1) * tw - inset
            tile = sheet[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            dest = os.path.join(dest_dir, f"{stem}_r{r}c{c}.jpg")
            cv2.imwrite(dest, tile)
            out.append(dest)
    return out


def download_test_images(force=False) -> list[dict]:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    ready = []
    for case in TEST_CASES:
        dest = os.path.join(IMAGE_DIR, case["name"])
        if os.path.isfile(dest) and os.path.getsize(dest) > 2000 and not force:
            ready.append({**case, "path": dest})
            print(f"[img] cached {case['name']}")
            continue

        ok = False
        if case.get("hf_file"):
            try:
                ok = _download_hf_file(case["hf_file"], dest)
            except Exception as exc:
                print(f"[img] HF failed {case['hf_file']}: {exc}")
        elif case.get("query"):
            try:
                url = _openverse_url(case["query"])
                if url:
                    print(f"[img] {case['name']} ← Openverse {url[:90]}")
                    ok = _save_image_bytes(dest, _http_get(url))
            except Exception as exc:
                print(f"[img] Openverse failed {case['name']}: {exc}")

        if not ok:
            print(f"[img] SKIP {case['name']}")
            continue
        ready.append({**case, "path": dest})
    return ready


def _download_pt(pt_path: str) -> str:
    from huggingface_hub import hf_hub_download

    from src.inference.traffic_sign_detector import HF_REPO, HF_WEIGHTS

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    print(f"[model] Downloading {HF_REPO}/{HF_WEIGHTS}")
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS)
    import shutil

    shutil.copy2(cached, pt_path)
    print(f"[model] Wrote {pt_path}")
    return pt_path


def run_images(detector: TrafficSignDetector, cases: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    annotated = []
    found_classes = set()
    for case in cases:
        frame = cv2.imread(case["path"])
        if frame is None:
            rows.append({**case, "ok": False, "dets": [], "note": "unreadable"})
            continue
        dets, speed = detector.detect(frame)
        vis = draw_detections(frame, dets)
        labels = [d["class"] for d in dets]
        found_classes.update(labels)
        expect = case.get("expect")
        hit = bool(labels) if expect is None else expect in labels
        out_path = os.path.join(out_dir, f"det_{case['name']}")
        cv2.imwrite(out_path, vis)
        annotated.append(vis)
        expect_s = expect or "any-class"
        print(
            f"[det] {case['name']:24s} expect={expect_s:20s} "
            f"{'HIT ' if hit else 'MISS'} {labels or '[]'} "
            f"infer={speed.get('inference', 0):.1f}ms"
        )
        rows.append(
            {
                "name": case["name"],
                "expect": expect_s,
                "hit": hit,
                "detections": [
                    {"class": d["class"], "conf": round(d["conf"], 3), "bbox": d["bbox"]}
                    for d in dets
                ],
                "speed_ms": {k: round(float(v), 2) for k, v in (speed or {}).items()},
                "output": out_path,
            }
        )
    if annotated:
        sheet = letterbox_stack(annotated, cols=4, cell=360)
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), sheet)
    return rows, found_classes


def benchmark(detector: TrafficSignDetector, image_path: str, warmup: int, iters: int):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    print(f"[bench] warmup={warmup} iters={iters} image={os.path.basename(image_path)}")
    for _ in range(warmup):
        detector.detect(frame)

    infer = []
    pre = []
    post = []
    wall = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _dets, speed = detector.detect(frame)
        wall.append((time.perf_counter() - t0) * 1000.0)
        infer.append(float(speed.get("inference", 0.0)))
        pre.append(float(speed.get("preprocess", 0.0)))
        post.append(float(speed.get("postprocess", 0.0)))

    def stats(xs):
        xs = sorted(xs)
        p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
        return {
            "mean_ms": round(statistics.mean(xs), 2),
            "median_ms": round(statistics.median(xs), 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(min(xs), 2),
            "max_ms": round(max(xs), 2),
        }

    report = {
        "model": detector.model_path,
        "imgsz": detector.imgsz,
        "conf": detector.conf_thresh,
        "warmup": warmup,
        "iters": iters,
        "image": image_path,
        "inference": stats(infer),
        "preprocess": stats(pre),
        "postprocess": stats(post),
        "end_to_end": stats(wall),
        "fps_from_infer": round(1000.0 / max(statistics.mean(infer), 1e-6), 1),
        "fps_end_to_end": round(1000.0 / max(statistics.mean(wall), 1e-6), 1),
    }
    return report


def write_report(path: str, detector: TrafficSignDetector, rows: list, found: set, bench: dict):
    hits = sum(1 for r in rows if r.get("hit"))
    lines = [
        "Traffic sign standalone test",
        f"model     : {detector.model_path}",
        f"imgsz/conf: {detector.imgsz} / {detector.conf_thresh}",
        f"images    : {hits}/{len(rows)} expected-class hits",
        f"classes hit in the wild: {', '.join(sorted(found)) or '(none)'}",
        "",
        "Per-image",
        "-" * 72,
    ]
    for r in rows:
        det_s = ", ".join(
            f"{d['class']} {d['conf']:.2f}" for d in r.get("detections", [])
        ) or "none"
        flag = "HIT " if r.get("hit") else "MISS"
        lines.append(f"{flag}  {r['name']:22s} expect={r['expect']:18s}  {det_s}")
    lines += [
        "",
        "Latency (Ultralytics timers + wall clock)",
        "-" * 72,
        json.dumps(bench, indent=2),
        "",
        "Missing expected classes: "
        + (
            ", ".join(
                sorted(
                    {
                        r["expect"]
                        for r in rows
                        if not r.get("hit") and r.get("expect") != "any-class"
                    }
                )
            )
            or "none"
        ),
        "Never detected at all: "
        + (
            ", ".join(c for c in TRAFFIC_SIGN_CLASSES if c not in found) or "none"
        ),
    ]
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--export-engine", action="store_true")
    parser.add_argument("--refresh-images", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.export_engine:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import export_traffic_sign_orin as _exp

        _exp.download_weights()
        onnx_path = _exp.export_onnx(args.imgsz)
        _exp.build_engine(onnx_path)

    model_path = args.model or resolve_model_path()
    if model_path is None:
        model_path = _download_pt(default_weight_paths()["pt"])

    print(f"[test] model={model_path}")
    detector = TrafficSignDetector(
        model_path=model_path, conf_thresh=args.conf, imgsz=args.imgsz
    )
    print(f"[test] names={detector.names}")

    cases = download_test_images(force=args.refresh_images)
    if not cases:
        print("[test] no images downloaded")
        return 1

    tile_dir = os.path.join(IMAGE_DIR, "tiles")
    tile_cases = []
    for case in cases:
        if not case.get("hf_file"):
            continue
        for tile in split_val_mosaic(case["path"], tile_dir):
            tile_cases.append(
                {
                    "name": os.path.basename(tile),
                    "expect": None,
                    "path": tile,
                }
            )
    if tile_cases:
        print(f"[test] split {len(tile_cases)} LISA val tiles from HF mosaics")
        cases = tile_cases + [c for c in cases if not c.get("hf_file")]

    rows, found = run_images(detector, cases, OUTPUT_DIR)
    bench_src = next((c["path"] for c in cases if "r0c0" in c["name"]), cases[0]["path"])
    bench = benchmark(detector, bench_src, args.warmup, args.iters)
    write_report(os.path.join(OUTPUT_DIR, "report.txt"), detector, rows, found, bench)
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"detections": rows, "latency": bench, "classes_found": sorted(found)}, f, indent=2)
    print(f"[test] wrote {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
