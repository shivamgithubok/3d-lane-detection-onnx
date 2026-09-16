#!/usr/bin/env python3
"""
Headless ADAS loop: lanes + vehicle YOLO + sign YOLO + LPRNet.

No Qt, no 15 FPS sleep. Reports compute latency and FPS on GRMN6693.

  python scripts/bench_isa_pipeline.py
  python scripts/bench_isa_pipeline.py --max-frames 1100
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

from src.inference.cipo_tracker import CIPOTracker
from src.inference.lprnet import LPRNetRecognizer, annotate_speed_dets
from src.inference.ldw_fcw import AdasWarningTracker, draw_adas_alerts
from src.inference.object_detector import OfflineYOLOVehicleDetector
from src.inference.postprocess import postprocess_onnx_output
from src.inference.speed_limit_tracker import SpeedLimitTracker
from src.inference.traffic_sign_detector import (
    TrafficSignDetector,
    draw_isa_overlay,
    filter_sign_dets,
    resolve_model_path,
)
from src.tracking.road_state import RoadStateEstimator
from src.utils.calibration import OPENLANE_CAM_HEIGHT, OPENLANE_CAM_PITCH_DEG, make_P_matrix
from src.utils.camera_transform import CameraTransform
from src.inference.lane_preprocess import prepare_lane_input
from src.utils.ego_speed import EgoSpeedLog
from src.utils.ground_calib import GroundCalibration
from src.utils.split_visualization import draw_front_view_cipo

INPUT_H, INPUT_W = 360, 480


def preprocess_frame(frame):
    frame_transform = CameraTransform.for_frame(frame, (INPUT_W, INPUT_H))
    resized = frame_transform.apply(frame)
    img, mask, meta = prepare_lane_input(resized)
    return img, mask, resized, frame_transform, meta


def _mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="testing_new_videos/GRMN6693_540_nohud.mp4")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--outdir", default="output/lprnet")
    args = ap.parse_args()

    video = args.video if os.path.isabs(args.video) else os.path.join(ROOT, args.video)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    if not os.path.isfile(video):
        print(f"Error: {video}")
        return 1

    import tensorrt as trt
    import pycuda.driver as cuda

    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    lpr = None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(os.path.join(ROOT, "models/anchor3dlane_raw.engine"), "rb") as f:
            with trt.Runtime(logger) as runtime:
                lane_engine = runtime.deserialize_cuda_engine(f.read())
        lane_ctx = lane_engine.create_execution_context()
        dummy_img = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        dummy_mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
        d_img = cuda.mem_alloc(dummy_img.nbytes)
        d_mask = cuda.mem_alloc(dummy_mask.nbytes)
        h_reg = np.empty((1, 4431, 86), dtype=np.float32)
        h_anc = np.empty((1, 4431, 65), dtype=np.float32)
        d_reg = cuda.mem_alloc(h_reg.nbytes)
        d_anc = cuda.mem_alloc(h_anc.nbytes)
        stream = cuda.Stream()
        lane_ctx.set_tensor_address("img", int(d_img))
        lane_ctx.set_tensor_address("mask", int(d_mask))
        lane_ctx.set_tensor_address("reg_proposals", int(d_reg))
        lane_ctx.set_tensor_address("anchors", int(d_anc))
        print("[bench] lane engine ready")

        cuda_ctx.pop()
        try:
            veh = OfflineYOLOVehicleDetector(
                model_path=os.path.join(ROOT, "models/yolov8n.engine"),
                conf_thresh=0.22,
                imgsz=640,
            )
            sign_path = resolve_model_path(os.path.join(ROOT, "models/traffic_sign_yolo11n.engine"))
            sign = TrafficSignDetector(model_path=sign_path, conf_thresh=0.35)
            print(f"[bench] sign engine {sign.model_path}")
        finally:
            cuda_ctx.push()

        lpr = LPRNetRecognizer(engine_path=os.path.join(ROOT, "models/us_lprnet_baseline18.engine"))
        print(f"[bench] LPRNet {lpr.engine_path}")

        P = make_P_matrix(OPENLANE_CAM_PITCH_DEG, OPENLANE_CAM_HEIGHT)
        gc = GroundCalibration.for_video(video)
        tracker = CIPOTracker(P_matrix=P, danger_dist=15.0, warning_dist=30.0, ground_calib=gc)
        road = RoadStateEstimator()
        isa = SpeedLimitTracker(confirm_hits=3, act_conf=0.55)
        warns = AdasWarningTracker()
        speed_log = EgoSpeedLog.auto_load(video)

        cap = cv2.VideoCapture(video)
        lane_ms, veh_ms, sign_ms, lpr_ms, e2e_ms = [], [], [], [], []
        last_dets = []
        events = []
        last_posted = None
        yolo_mph_counts = {}
        lpr_mph_counts = {}
        ldw_n = fcw_n = fcw_plus_n = 0
        alert_stills = 0
        stills = 0
        i = 0
        t_all = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and i >= args.max_frames:
                break
            t0 = time.perf_counter()

            img, mask, _, frame_tf, prep = preprocess_frame(frame)
            t_lane0 = time.perf_counter()
            cuda.memcpy_htod_async(d_img, img, stream)
            cuda.memcpy_htod_async(d_mask, mask, stream)
            lane_ctx.execute_async_v3(stream.handle)
            cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
            cuda.memcpy_dtoh_async(h_anc, d_anc, stream)
            stream.synchronize()
            lane_ms.append((time.perf_counter() - t_lane0) * 1000.0)
            raw, _ = postprocess_onnx_output(h_reg, conf_threshold=prep["conf"])
            speed_mps = speed_log.get_mps(i, min_mps=0.0) if speed_log else None
            st = road.update(raw, dt=1.0 / 30.0, speed_mps=speed_mps)

            cuda_ctx.pop()
            ran_sign = False
            dets = last_dets
            try:
                t_v = time.perf_counter()
                veh_dets = veh.detect(frame)
                veh_ms.append((time.perf_counter() - t_v) * 1000.0)
                if (i % max(1, args.every)) == 0:
                    t_s = time.perf_counter()
                    dets, _ = sign.detect(frame)
                    dets = filter_sign_dets(dets, frame.shape)
                    sign_ms.append((time.perf_counter() - t_s) * 1000.0)
                    for d in dets:
                        yc = d.get("class")
                        yolo_mph_counts[yc] = yolo_mph_counts.get(yc, 0) + 1
                    last_dets = dets
                    ran_sign = True
            finally:
                cuda_ctx.push()

            if ran_sign:
                t_l = time.perf_counter()
                if dets:
                    annotate_speed_dets(frame, dets, lpr)
                lpr_ms.append((time.perf_counter() - t_l) * 1000.0)
                for d in dets:
                    if d.get("mph") is not None:
                        lpr_mph_counts[d["mph"]] = lpr_mph_counts.get(d["mph"], 0) + 1
                snap = isa.update(dets, ran_infer=True, use_class=False)
            else:
                snap = isa.update([], ran_infer=False, use_class=False)

            if snap["posted_mph"] != last_posted and snap["status"] == "CONFIRMED":
                events.append(
                    {
                        "frame": i,
                        "mph": snap["posted_mph"],
                        "conf": round(float(snap["conf"]), 3),
                    }
                )
                last_posted = snap["posted_mph"]
                print(f"[isa] CONFIRMED {snap['posted_mph']} mph @ frame {i}", flush=True)

            h, w = frame.shape[:2]
            tracker.ground_calib = gc.adapted_to(w, h) if gc is not None else gc
            processed, cipo = tracker.process_detections(
                veh_dets,
                st.lanes,
                frame_size=(w, h),
                ego_left=st.ego_left,
                ego_right=st.ego_right,
                frame_transform=frame_tf,
                road_state_confirmed=st.is_confirmed,
                road_status=st.status,
                left_corridor_3d=st.left_corridor_3d,
                right_corridor_3d=st.right_corridor_3d,
                dt=1.0 / 30.0,
                ego_speed_mps=speed_mps,
            )
            cipo_status = tracker.last_cipo_status
            alerts = warns.update(
                st.ego_left, st.ego_right, st.status, cipo, cipo_status, speed_mps, dt=1.0 / 30.0
            )
            if alerts["ldw"] in ("LEFT", "RIGHT"):
                ldw_n += 1
            if alerts["fcw"] == "FCW":
                fcw_n += 1
            elif alerts["fcw"] == "FCW+":
                fcw_plus_n += 1

            if not args.no_draw:
                vis = draw_front_view_cipo(
                    frame,
                    st.visual_lanes,
                    processed,
                    cipo,
                    np.asarray(P, dtype=np.float64),
                    show_drivable=True,
                    ego_left=st.ego_left,
                    ego_right=st.ego_right,
                    frame_transform=frame_tf,
                    road_state_valid=(st.left_corridor_3d is not None),
                    left_corridor_3d=st.left_corridor_3d,
                    right_corridor_3d=st.right_corridor_3d,
                )
                ego = speed_log.get_mph(i) if speed_log else None
                vis = draw_isa_overlay(vis, snap, ego_mph=ego, detections=dets)
                vis = draw_adas_alerts(
                    vis,
                    alerts,
                    ego_left=st.ego_left,
                    ego_right=st.ego_right,
                    P_matrix=np.asarray(P, dtype=np.float64),
                    frame_transform=frame_tf,
                    cipo_obj=cipo,
                )
                if ran_sign and dets and stills < 8 and (
                    snap.get("posted_mph") is not None or snap.get("candidate_mph") is not None
                ):
                    cv2.imwrite(os.path.join(outdir, f"pipeline_f{i:04d}.jpg"), vis)
                    stills += 1
                if alerts["priority"] != "none" and alert_stills < 16:
                    cv2.imwrite(os.path.join(outdir, f"alert_f{i:04d}.jpg"), vis)
                    alert_stills += 1

            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
            i += 1
            if i % 200 == 0:
                print(
                    f"[bench] f={i} e2e={_mean(e2e_ms[-30:]):.1f}ms "
                    f"posted={snap['posted_mph']} {snap['status']}",
                    flush=True,
                )

        cap.release()
        elapsed = time.perf_counter() - t_all
        stats = {
            "video": video,
            "frames": i,
            "sign_every": args.every,
            "elapsed_s": round(elapsed, 2),
            "compute_fps": round(i / max(elapsed, 1e-6), 2),
            "e2e_mean_ms": round(_mean(e2e_ms), 2),
            "lane_mean_ms": round(_mean(lane_ms), 2),
            "vehicle_yolo_mean_ms": round(_mean(veh_ms), 2),
            "sign_yolo_mean_ms": round(_mean(sign_ms), 2),
            "lprnet_mean_ms": round(_mean(lpr_ms), 2),
            "sign_infer_frames": len(sign_ms),
            "lpr_frames": len(lpr_ms),
            "confirmed": [e["mph"] for e in events],
            "events": events,
            "yolo_class_counts": yolo_mph_counts,
            "lpr_mph_counts": {str(k): v for k, v in lpr_mph_counts.items()},
            "ldw_frames": ldw_n,
            "fcw_frames": fcw_n,
            "fcw_plus_frames": fcw_plus_n,
            "draw": not args.no_draw,
            "ui_pace_note": "ADAS UI still sleeps to ~15 FPS; these numbers are uncapped compute.",
        }
        txt = os.path.join(outdir, "pipeline_bench.txt")
        js = os.path.join(outdir, "pipeline_bench.json")
        lines = [
            "ADAS pipeline + LPRNet (headless, no 15 FPS cap)",
            f"video     : {video}",
            f"frames    : {i}  in {elapsed:.1f}s",
            f"compute   : {stats['e2e_mean_ms']:.1f} ms/frame  ({stats['compute_fps']:.1f} FPS)",
            f"  lane           {stats['lane_mean_ms']:.2f} ms",
            f"  vehicle YOLO   {stats['vehicle_yolo_mean_ms']:.2f} ms",
            f"  sign YOLO      {stats['sign_yolo_mean_ms']:.2f} ms  (every {args.every})",
            f"  LPRNet OCR     {stats['lprnet_mean_ms']:.2f} ms  (on sign frames)",
            f"ISA confirmed    {stats['confirmed'] or 'none'}",
            f"YOLO classes     {yolo_mph_counts}",
            f"LPRNet mph       {lpr_mph_counts}",
            f"LDW frames       {ldw_n}",
            f"FCW / FCW+       {fcw_n} / {fcw_plus_n}",
            "UI note: worker still paces ~15 FPS with msleep; HUD latency is e2e_mean above.",
        ]
        text = "\n".join(lines) + "\n"
        with open(txt, "w", encoding="utf-8") as f:
            f.write(text)
        with open(js, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print("\n" + text)
        print(f"[wrote] {txt}")
        return 0
    finally:
        if lpr is not None:
            try:
                lpr.close()
            except Exception:
                pass
        try:
            cuda_ctx.pop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
