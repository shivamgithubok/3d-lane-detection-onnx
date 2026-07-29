import cv2
import numpy as np
import time
import sys
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels, ANCHOR_Y_STEPS
from src.utils.visualization import draw_bev
from src.tracking.lane_association import LaneTrackerManager

ENGINE_PATH = "models/anchor3dlane_raw.engine"
DEFAULT_VIDEO_PATH = "data/images/example_2.mp4"
OUTPUT_VIDEO_PATH = "output/example_2_ekf_annotated.mp4"

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

def project_3d_points(pts_3d, P_matrix):
    xs = pts_3d[:, 0]
    ys = pts_3d[:, 1]
    zs = pts_3d[:, 2]
    ones = np.ones((1, len(zs)))
    coords = np.vstack((xs, ys, zs, ones))
    trans = P_matrix @ coords
    u = trans[0, :] / (trans[2, :] + 1e-8)
    v = trans[1, :] / (trans[2, :] + 1e-8)
    return np.column_stack((u, v))

def draw_lanes(frame, proposals, color=(0, 255, 0), thickness=2):
    if proposals is None or len(proposals) == 0:
        return frame
    for lane in proposals:
        if isinstance(lane, np.ndarray) and lane.ndim == 2 and lane.shape[1] == 3:
            pts = project_3d_points(lane, P_MATRIX)
        else:
            pts = decode_lane_pixels(lane, P_MATRIX)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]]
        for i in range(1, len(draw_pts)):
            cv2.line(frame, draw_pts[i-1], draw_pts[i], color, thickness)
    return frame

def run_ekf_video_inference(video_path=DEFAULT_VIDEO_PATH, save_output=True):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"Loading TensorRT Engine from {ENGINE_PATH}...")
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

    # Initialize EKF Multi-Lane Tracker Manager
    tracker_manager = LaneTrackerManager(max_missed_frames=10, dist_threshold=2.5)

    print(f"Opening video source: {video_path}")
    cap = cv2.VideoCapture(video_path)

    os.makedirs("output", exist_ok=True)
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (960, 720))

    print("Starting TensorRT Real-Time EKF Video Inference...")
    frame_count = 0
    fps_history = []

    while cap.isOpened():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        img, mask, resized = preprocess(frame)
        h_img = np.ascontiguousarray(img)
        h_mask = np.ascontiguousarray(mask)

        cuda.memcpy_htod_async(d_img, h_img, stream)
        cuda.memcpy_htod_async(d_mask, h_mask, stream)

        context.execute_async_v3(stream.handle)

        cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
        cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
        stream.synchronize()

        # 1. Raw neural network proposals
        raw_proposals, scores = postprocess_onnx_output(h_reg_proposals)

        # 2. Update EKF Temporal Tracker
        dt = 0.033
        smoothed_ekf_lanes = tracker_manager.update(raw_proposals, dt=dt)

        t_end = time.perf_counter()

        dt_total = t_end - t_start
        fps = 1.0 / dt_total if dt_total > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history[-30:])

        # Draw EKF smoothed lanes in Cyan (0, 255, 255)
        front_annotated = draw_lanes(resized.copy(), smoothed_ekf_lanes, color=(255, 255, 0), thickness=3)
        bev = draw_bev(smoothed_ekf_lanes, ANCHOR_Y_STEPS)

        display_frame = cv2.resize(front_annotated, (960, 720))
        cv2.putText(display_frame, f"GPU EKF FPS: {avg_fps:.1f} ({dt_total*1000:.1f}ms)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        num_lanes = len(smoothed_ekf_lanes)
        cv2.putText(display_frame, f"Active EKF Tracks: {num_lanes}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(display_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | Live GPU EKF FPS: {avg_fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print(f"\nCompleted! Processed {frame_count} frames.")
    print(f"Average EKF Pipeline FPS: {np.mean(fps_history):.2f} FPS")
    if save_output:
        print(f"Annotated EKF video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    video_input = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    run_ekf_video_inference(video_input)
