"""Compare sky-crop vs OpenLane road warp, then a paint snap. Saves overlays."""

import os
import sys

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.lane_preprocess import prepare_lane_input
from src.inference.postprocess import decode_lane_pixels, postprocess_onnx_output
from src.ui.worker import preprocess_frame
from src.utils.calibration import P_final
from src.utils.camera_transform import CameraTransform
from src.utils.ground_calib import GroundCalibration
from src.utils.openlane_view import OpenLaneView

VIDEO = "testing_new_videos/GRMN6694_540_nohud.mp4"
ENGINE = "models/anchor3dlane_raw.engine"
FRAMES = [400, 900, 1035, 1200]
OUT = "output/openlane_view_check"


def draw_lanes(frame, proposals, P, transform, color=(0, 255, 0)):
    out = frame.copy()
    h, w = out.shape[:2]
    for lane in proposals:
        pts = decode_lane_pixels(lane, P, flat_ground=False)
        model = np.asarray([(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360], dtype=np.float64)
        if len(model) < 2:
            continue
        src = transform.model_to_source(model)
        poly = []
        for u, v in src:
            if 0 <= u < w and 0 <= v < h:
                poly.append((int(round(u)), int(round(v))))
        if len(poly) >= 2:
            cv2.polylines(out, [np.asarray(poly, np.int32)], False, color, 2, cv2.LINE_AA)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        d_img = cuda.mem_alloc(1 * 3 * 360 * 480 * 4)
        d_mask = cuda.mem_alloc(1 * 1 * 360 * 480 * 4)
        h_reg = np.empty((1, 4431, 86), np.float32)
        h_anc = np.empty((1, 4431, 65), np.float32)
        d_reg = cuda.mem_alloc(h_reg.nbytes)
        d_anc = cuda.mem_alloc(h_anc.nbytes)
        stream = cuda.Stream()
        context.set_tensor_address("img", int(d_img))
        context.set_tensor_address("mask", int(d_mask))
        context.set_tensor_address("reg_proposals", int(d_reg))
        context.set_tensor_address("anchors", int(d_anc))

        cap = cv2.VideoCapture(VIDEO)
        ok, frame0 = cap.read()
        calib = GroundCalibration.for_video(VIDEO, frame0.shape[:2])
        view = OpenLaneView(calib, P_final)
        print("calib", calib.source, "f", round(calib.f_px, 1), "h", round(calib.cam_height_m, 3))

        for fi in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                print("miss", fi)
                continue
            crop_tf = CameraTransform.for_frame(frame, (480, 360))
            _, _, crop_img, _, meta_c = preprocess_frame(frame, None)
            _, _, warp_img, _, meta_w = preprocess_frame(frame, view)

            def infer(img, mask, conf):
                cuda.memcpy_htod_async(d_img, img, stream)
                cuda.memcpy_htod_async(d_mask, mask, stream)
                context.execute_async_v3(stream.handle)
                cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
                stream.synchronize()
                props, _ = postprocess_onnx_output(h_reg, conf_threshold=conf)
                return props

            img_c, mask_c, _ = prepare_lane_input(crop_tf.apply(frame))
            img_w, mask_w, meta_w = prepare_lane_input(view.apply(frame))
            props_c = infer(img_c, mask_c, meta_c["conf"] if False else 0.43)
            props_w = infer(img_w, mask_w, meta_w["conf"])
            print(f"frame {fi}: crop lanes {len(props_c)}  warp lanes {len(props_w)}")

            vis_c = draw_lanes(frame, props_c, P_final, crop_tf, (0, 180, 255))
            vis_w = draw_lanes(frame, props_w, P_final, view, (0, 255, 0))
            cv2.imwrite(f"{OUT}/f{fi}_crop.jpg", vis_c)
            cv2.imwrite(f"{OUT}/f{fi}_warp.jpg", vis_w)
            cv2.imwrite(f"{OUT}/f{fi}_model_crop.jpg", crop_tf.apply(frame))
            cv2.imwrite(f"{OUT}/f{fi}_model_warp.jpg", view.apply(frame))
            # lanes drawn in the model image itself
            model = view.apply(frame)
            for lane in props_w:
                pts = decode_lane_pixels(lane, P_final, flat_ground=False)
                poly = [(int(round(u)), int(round(v))) for u, v in pts if 0 <= u < 480 and 0 <= v < 360]
                if len(poly) >= 2:
                    cv2.polylines(model, [np.asarray(poly, np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(f"{OUT}/f{fi}_model_warp_lanes.jpg", model)
        cap.release()
    finally:
        ctx.pop()


if __name__ == "__main__":
    main()
