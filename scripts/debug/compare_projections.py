import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels

ENGINE_PATH = "models/anchor3dlane_raw.engine"
VIDEO_PATH = "data/images/example_2.mp4"
INPUT_H, INPUT_W = 360, 480
IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# OpenLane Ground-Truth Projection Matrix
P_OPENLANE = np.array([
    [517.368023818057, -1.2192886414581698, 245.2892094158112, -1317.8869498335648],
    [-2.080056338959641, 584.7261604964109, 187.72380574460553, -379.84979423293373],
    [-0.012627197281367759, -0.004025390168358267, 0.9999121712044562, -2.096266740239456]
])

def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    return img, mask, resized

def main():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    reg_proposals_shape = (1, 4431, 86)
    anchors_shape = (1, 4431, 65)

    dummy_img = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
    dummy_mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    
    d_img = cuda.mem_alloc(dummy_img.nbytes)
    d_mask = cuda.mem_alloc(dummy_mask.nbytes)
    h_reg_proposals = np.empty(reg_proposals_shape, dtype=np.float32)
    h_anchors = np.empty(anchors_shape, dtype=np.float32)
    d_reg_proposals = cuda.mem_alloc(h_reg_proposals.nbytes)
    d_anchors = cuda.mem_alloc(h_anchors.nbytes)

    stream = cuda.Stream()
    context.set_tensor_address("img", int(d_img))
    context.set_tensor_address("mask", int(d_mask))
    context.set_tensor_address("reg_proposals", int(d_reg_proposals))
    context.set_tensor_address("anchors", int(d_anchors))

    cap = cv2.VideoCapture(VIDEO_PATH)
    for _ in range(80):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    img, mask, resized = preprocess(frame)
    h_img = np.ascontiguousarray(img)
    h_mask = np.ascontiguousarray(mask)

    cuda.memcpy_htod_async(d_img, h_img, stream)
    cuda.memcpy_htod_async(d_mask, h_mask, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
    cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
    stream.synchronize()

    proposals, scores = postprocess_onnx_output(h_reg_proposals, conf_threshold=0.2)

    print(f"Detected {len(proposals)} lane proposals.")

    # Calculate pixel range with P_OPENLANE
    vs_openlane = []
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_OPENLANE)
        for u, v in pts:
            if 0 <= u < INPUT_W and 0 <= v < INPUT_H:
                vs_openlane.append(v)

    if vs_openlane:
        print(f"OpenLane P Matrix pixel y (v) range: {min(vs_openlane):.1f} to {max(vs_openlane):.1f} (Frame height: {INPUT_H})")

    # Draw overlay with P_OPENLANE
    canvas = resized.copy()
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_OPENLANE)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < INPUT_W and 0 <= v < INPUT_H]
        for i in range(1, len(draw_pts)):
            cv2.line(canvas, draw_pts[i-1], draw_pts[i], (0, 255, 0), 2)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/openlane_calibration_test.jpg", canvas)
    print("Saved output/openlane_calibration_test.jpg")

if __name__ == "__main__":
    main()
