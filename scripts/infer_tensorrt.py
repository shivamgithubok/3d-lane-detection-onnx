import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels

ENGINE_PATH = "models/anchor3dlane_raw.engine"
IMAGE_PATH = "data/images/video_frame_60.jpg" if os.path.exists("data/images/video_frame_60.jpg") else "data/images/example.jpg"
OUTPUT_PATH = "output/tensorrt_annotated.jpg"

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

P_MATRIX = np.array([
    [503.75, 239.67108834, 12.5606295, 0.0],
    [0.0, 181.326628, -557.993558, 850.078125],
    [0.0, 0.998629535, 0.0523359562, 0.0]
])

def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    return img, mask, resized

def draw_lanes(frame, proposals):
    if proposals is None:
        return frame
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_MATRIX)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]]
        for i in range(1, len(draw_pts)):
            cv2.line(frame, draw_pts[i-1], draw_pts[i], (0, 0, 255), 2)
    return frame

def run_tensorrt_inference():
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"Loading TensorRT Engine from {ENGINE_PATH}...")
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    print(f"Reading image from {IMAGE_PATH}...")
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not read image at {IMAGE_PATH}")
        return

    img, mask, resized = preprocess(frame)

    # Allocate CUDA memory buffers for inputs and outputs
    h_img = np.ascontiguousarray(img)
    h_mask = np.ascontiguousarray(mask)
    
    d_img = cuda.mem_alloc(h_img.nbytes)
    d_mask = cuda.mem_alloc(h_mask.nbytes)
    
    # Model output shapes matching TensorRT engine contract
    reg_proposals_shape = (1, 4431, 86)
    anchors_shape = (1, 4431, 65)
    
    h_reg_proposals = np.empty(reg_proposals_shape, dtype=np.float32)
    h_anchors = np.empty(anchors_shape, dtype=np.float32)
    
    d_reg_proposals = cuda.mem_alloc(h_reg_proposals.nbytes)
    d_anchors = cuda.mem_alloc(h_anchors.nbytes)

    stream = cuda.Stream()

    # Transfer input data to GPU
    cuda.memcpy_htod_async(d_img, h_img, stream)
    cuda.memcpy_htod_async(d_mask, h_mask, stream)

    context.set_tensor_address("img", int(d_img))
    context.set_tensor_address("mask", int(d_mask))
    context.set_tensor_address("reg_proposals", int(d_reg_proposals))
    context.set_tensor_address("anchors", int(d_anchors))

    # Warmup
    for _ in range(10):
        context.execute_async_v3(stream.handle)
    stream.synchronize()

    # Benchmark loop
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
        cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
        stream.synchronize()
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)

    proposals, scores = postprocess_onnx_output(h_reg_proposals)

    avg_ms = np.mean(latencies)
    fps = 1000.0 / avg_ms
    num_lanes = 0 if proposals is None else len(proposals)

    print(f"\n================ TENSORRT INFERENCE ================")
    print(f"GPU Inference Time : {avg_ms:.2f} ms")
    print(f"GPU Throughput     : {fps:.2f} FPS")
    print(f"Detections         : Found {num_lanes} lanes")
    print("====================================================\n")

    os.makedirs("output", exist_ok=True)
    annotated = draw_lanes(resized.copy(), proposals)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Annotated result saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_tensorrt_inference()
