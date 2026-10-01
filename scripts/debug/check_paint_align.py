"""Render the paint-snapped front overlay on a few Garmin frames."""

import os
import sys

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.lane_preprocess import prepare_lane_input
from src.inference.postprocess import postprocess_onnx_output
from src.utils.calibration import P_final
from src.utils.drivable_area import find_ego_lanes
from src.utils.ground_calib import GroundCalibration
from src.utils.openlane_view import OpenLaneView
from src.utils.paint_snap import lane_paint_mask, snap_hit_fraction
from src.utils.split_visualization import draw_front_view_cipo

VIDEO = "testing_new_videos/GRMN6694_540_nohud.mp4"
FRAMES = [400, 900, 1035, 1200]
OUT = "output/paint_snap_check"


def paint_distance(frame, proposals, view, calib):
    """Median pixels from each projected lane sample to the nearest paint pixel."""
    from src.inference.postprocess import decode_lane_pixels
    from src.utils.paint_snap import snap_polyline_to_paint

    mask = lane_paint_mask(frame)
    h, w = mask.shape
    # distance transform wants non-paint as foreground for distance-to-paint
    inv = (mask == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    before, after = [], []
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_final, flat_ground=False)
        model = np.asarray([(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360], np.float64)
        if len(model) < 2:
            continue
        src = view.model_to_source(model)
        raw = []
        for u, v in src:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and int(0.45 * h) <= vi <= int(0.92 * h):
                raw.append((ui, vi))
                before.append(float(dist[vi, ui]))
        snapped = snap_polyline_to_paint(raw, frame, calib, mask=mask)
        for u, v in snapped:
            if 0 <= u < w and 0 <= v < h:
                after.append(float(dist[v, u]))
    def med(xs):
        return float(np.median(xs)) if xs else float("nan")
    return med(before), med(after), len(before)


def main():
    os.makedirs(OUT, exist_ok=True)
    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(open("models/anchor3dlane_raw.engine", "rb").read())
        context = engine.create_execution_context()
        d_img = cuda.mem_alloc(1 * 3 * 360 * 480 * 4)
        d_mask = cuda.mem_alloc(1 * 1 * 360 * 480 * 4)
        h_reg = np.empty((1, 4431, 86), np.float32)
        d_reg = cuda.mem_alloc(h_reg.nbytes)
        d_anc = cuda.mem_alloc(np.empty((1, 4431, 65), np.float32).nbytes)
        stream = cuda.Stream()
        context.set_tensor_address("img", int(d_img))
        context.set_tensor_address("mask", int(d_mask))
        context.set_tensor_address("reg_proposals", int(d_reg))
        context.set_tensor_address("anchors", int(d_anc))

        cap = cv2.VideoCapture(VIDEO)
        ok, frame0 = cap.read()
        calib = GroundCalibration.for_video(VIDEO, frame0.shape[:2])
        view = OpenLaneView(calib, P_final)
        print("calib", calib.source)

        for fi in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                print("miss", fi)
                continue
            warped = view.apply(frame)
            img, mask, meta = prepare_lane_input(warped)
            cuda.memcpy_htod_async(d_img, img, stream)
            cuda.memcpy_htod_async(d_mask, mask, stream)
            context.execute_async_v3(stream.handle)
            cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
            stream.synchronize()
            props, _ = postprocess_onnx_output(h_reg, conf_threshold=meta["conf"])
            ego_l, ego_r = find_ego_lanes(props)
            vis = draw_front_view_cipo(
                frame, props, [], None, P_final,
                show_drivable=True,
                ego_left=ego_l, ego_right=ego_r,
                frame_transform=view,
                road_state_valid=ego_l is not None and ego_r is not None,
                ground_calib=calib,
            )
            b, a, n = paint_distance(frame, props, view, calib)
            from src.inference.postprocess import decode_lane_pixels as _dec
            pm = lane_paint_mask(frame, calib)
            fracs = []
            for lane in props:
                pts = _dec(lane, P_final, flat_ground=False)
                model = [(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360]
                if len(model) < 2:
                    continue
                src = view.model_to_source(np.asarray(model, np.float64))
                raw = [(float(u), float(v)) for u, v in src]
                fracs.append(snap_hit_fraction(raw, frame, calib, mask=pm))
            print(
                f"frame {fi}: lanes {len(props)} samples {n}  "
                f"paint-px before {b:.1f} after {a:.1f}  hit {['%.2f'%x for x in fracs]}"
            )
            cv2.imwrite(f"{OUT}/f{fi}.jpg", vis)
            dbg = frame.copy()
            pm = lane_paint_mask(frame)
            dbg[pm > 0] = (255, 0, 0)
            from src.inference.postprocess import decode_lane_pixels
            from src.utils.paint_snap import snap_polyline_to_paint
            for lane in props:
                pts = decode_lane_pixels(lane, P_final, flat_ground=False)
                model = np.asarray([(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360], np.float64)
                if len(model) < 2:
                    continue
                src = view.model_to_source(model)
                raw = [(int(round(u)), int(round(v))) for u, v in src]
                snapped = snap_polyline_to_paint(raw, frame, calib, mask=pm)
                for u, v in raw:
                    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                        cv2.circle(dbg, (u, v), 3, (0, 0, 255), -1)
                for u, v in snapped:
                    cv2.circle(dbg, (u, v), 3, (0, 255, 0), -1)
            cv2.imwrite(f"{OUT}/f{fi}_dbg.jpg", dbg)
        cap.release()
    finally:
        ctx.pop()


if __name__ == "__main__":
    main()
