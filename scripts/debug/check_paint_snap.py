"""Shift each warped lane onto the nearest paint and save the overlay."""

import os
import sys

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.lane_preprocess import prepare_lane_input
from src.inference.postprocess import ANCHOR_LEN, decode_lane_pixels, postprocess_onnx_output
from src.utils.calibration import P_final
from src.utils.ground_calib import GroundCalibration
from src.utils.openlane_view import OpenLaneView

VIDEO = "testing_new_videos/GRMN6694_540_nohud.mp4"
FRAMES = [400, 900, 1035, 1200]
OUT = "output/openlane_view_check"


def paint_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 75)
    yellow = (
        (hsv[:, :, 0] >= 18)
        & (hsv[:, :, 0] <= 38)
        & (hsv[:, :, 1] > 90)
        & (hsv[:, :, 2] > 120)
    )
    m = (white | yellow).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def nearest_paint_dx(calib, mask, u, v, window_m=1.2):
    hit = calib.backproject_ground(float(u), float(v))
    if hit is None:
        return None
    x, y = hit
    du = calib.f_px * window_m / max(y, 1.0)
    v = int(round(v))
    u = int(round(u))
    if not (0 <= v < mask.shape[0]):
        return None
    u0 = max(0, int(u - du))
    u1 = min(mask.shape[1] - 1, int(u + du))
    cols = np.where(mask[v, u0 : u1 + 1] > 0)[0]
    if len(cols) < 2:
        return None
    cols = cols + u0
    groups = []
    cur = [cols[0]]
    for c in cols[1:]:
        if c - cur[-1] <= 8:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    best = None
    for g in groups:
        if len(g) < 2:
            continue
        hw = calib.backproject_ground(float(np.median(g)), float(v))
        if hw is None:
            continue
        dx = hw[0] - x
        if best is None or abs(dx) < abs(best):
            best = dx
    return best


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
        for fi in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            calib = GroundCalibration.for_video(VIDEO, frame.shape[:2])
            view = OpenLaneView(calib, P_final)
            img, mask, _meta = prepare_lane_input(view.apply(frame))
            cuda.memcpy_htod_async(d_img, img, stream)
            cuda.memcpy_htod_async(d_mask, mask, stream)
            context.execute_async_v3(stream.handle)
            cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
            stream.synchronize()
            props, _ = postprocess_onnx_output(h_reg, 0.43)
            pm = paint_mask(frame)
            vis = frame.copy()
            print(f"frame {fi} lanes {len(props)}")
            for li, p in enumerate(props):
                pts = decode_lane_pixels(p, P_final, flat_ground=True)
                model = np.asarray(
                    [(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360], np.float64
                )
                if len(model) < 2:
                    continue
                src = view.model_to_source(model)
                dxs = []
                for u, v in src:
                    dx = nearest_paint_dx(calib, pm, u, v)
                    if dx is not None:
                        dxs.append(dx)
                med = float(np.median(dxs)) if dxs else 0.0
                print(f"  lane {li} n={len(dxs)} dx {med:+.2f} samples {np.round(dxs, 2)}")
                q = np.array(p, copy=True)
                q[5 : 5 + ANCHOR_LEN] = q[5 : 5 + ANCHOR_LEN] + med
                pts2 = decode_lane_pixels(q, P_final, flat_ground=True)
                model2 = np.asarray(
                    [(u, v) for u, v in pts2 if 0 <= u < 480 and 0 <= v < 360], np.float64
                )
                src2 = view.model_to_source(model2)
                poly = [
                    (int(round(u)), int(round(v)))
                    for u, v in src2
                    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]
                ]
                if len(poly) >= 2:
                    cv2.polylines(vis, [np.asarray(poly, np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(f"{OUT}/f{fi}_snap.jpg", vis)
        cap.release()
    finally:
        ctx.pop()


if __name__ == "__main__":
    main()
