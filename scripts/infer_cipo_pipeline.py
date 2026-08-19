import cv2
import numpy as np
import time
import sys
import os
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels
from src.inference.cipo_tracker import CIPOTracker, DEFAULT_P_MATRIX
from src.inference.object_detector import OfflineYOLOVehicleDetector
from src.inference.trt_depth_estimator import TRTMonocularDepthEstimator
from src.utils.split_visualization import draw_bev_cipo, draw_front_view_cipo, create_split_window
from src.utils.camera_transform import CameraTransform
from src.tracking.road_state import RoadStateEstimator

ENGINE_PATH = "models/anchor3dlane_raw.engine"
DEFAULT_VIDEO_PATH = "data/images/example_3.mp4"
OUTPUT_VIDEO_PATH = "output/example_3_futuristic_adas.mp4"

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

def preprocess(frame):
    frame_transform = CameraTransform.for_frame(frame, (INPUT_W, INPUT_H))
    resized = frame_transform.apply(frame)
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    img = np.ascontiguousarray(img)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    mask = np.ascontiguousarray(mask)
    return img, mask, resized, frame_transform

def run_cipo_pipeline(video_path=DEFAULT_VIDEO_PATH, output_path=OUTPUT_VIDEO_PATH, show_gui=True, max_frames=None, show_drivable=True):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return

    print("================================================================")
    print(" 🛣️  3D LANE + DRIVABLE AREA + YOLO CIPO (Futuristic ADAS UI)")
    print("================================================================")

    # 1. Initialize TensorRT Engine for Anchor3DLane
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"[TRT] Loading 3D Lane TensorRT Engine from: {ENGINE_PATH}")
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

    # 2. Initialize Object Detector & CIPO Tracker
    print("[Pipeline] Initializing YOLO-nano ByteTrack, lanes & depth engines...")
    detector = OfflineYOLOVehicleDetector(
        model_path="models/yolov8n.engine", conf_thresh=0.22, imgsz=640
    )
    depth_estimator = TRTMonocularDepthEstimator("models/monocular_depth.engine")
    tracker = CIPOTracker(P_matrix=DEFAULT_P_MATRIX, danger_dist=15.0, warning_dist=30.0)
    road_state_estimator = RoadStateEstimator()

    # 3. Open Video Source
    print(f"[Video] Opening video source: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1080, 720))

    window_name = "Real-Time 3D Lane + Drivable Area + YOLO CIPO ADAS Pipeline"
    if show_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1080, 720)
        except Exception:
            show_gui = False

    frame_idx = 0
    fps_history = []
    start_total_time = time.time()

    print("[Pipeline] Starting inference loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t0 = time.time()

        # Step A: Preprocess Frame for 3D Lane Model
        img_tensor, mask_tensor, resized_frame, frame_transform = preprocess(frame)

        # Step B: TensorRT GPU Inference for 3D Lanes
        cuda.memcpy_htod_async(d_img, img_tensor, stream)
        cuda.memcpy_htod_async(d_mask, mask_tensor, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
        cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
        stream.synchronize()

        # Step C: Decode 3D Lane Proposals
        raw_lane_proposals, lane_scores = postprocess_onnx_output(h_reg_proposals)
        road_state = road_state_estimator.update(raw_lane_proposals, dt=1.0 / source_fps)
        visual_lanes = road_state.visual_lanes

        # Step D: YOLO (release pycuda ctx so Ultralytics TRT can run) + depth
        yolo_ctx = None
        try:
            yolo_ctx = cuda.Context.get_current()
            if yolo_ctx is not None:
                yolo_ctx.pop()
        except Exception:
            yolo_ctx = None
        try:
            raw_detections = detector.detect(frame)
        finally:
            if yolo_ctx is not None:
                try:
                    yolo_ctx.push()
                except Exception:
                    pass
        depth_map, _, _ = depth_estimator.estimate_depth_map(frame)

        # Step E: Process CIPO Tracker & 3D ROI In-Path Check
        h_frame, w_frame = frame.shape[:2]
        processed_objs, cipo_obj = tracker.process_detections(
            raw_detections, 
            road_state.lanes,
            frame_size=(w_frame, h_frame),
            depth_map=depth_map,
            depth_estimator=depth_estimator,
            ego_left=road_state.ego_left,
            ego_right=road_state.ego_right,
            frame_transform=frame_transform,
            road_state_confirmed=road_state.is_confirmed,
        )

        t1 = time.time()
        frame_time = t1 - t0
        fps = 1.0 / max(0.001, frame_time)
        fps_history.append(fps)

        if cipo_obj is not None:
            cipo_status = "DANGER" if cipo_obj["Z_3d"] < tracker.danger_dist else (
                "WARNING" if cipo_obj["Z_3d"] < tracker.warning_dist else "SAFE"
            )
            cipo_obj["status"] = cipo_status
        elif road_state.status != "CONFIRMED":
            cipo_status = "DEGRADED"
        else:
            cipo_status = "SAFE"

        # Step F: Render 3-Panel Split Window (Front View + BEV + HUD)
        front_view = draw_front_view_cipo(
            frame,
            visual_lanes,
            processed_objs,
            cipo_obj,
            DEFAULT_P_MATRIX,
            show_drivable=show_drivable and road_state.has_valid_corridor,
            ego_left=road_state.ego_left,
            ego_right=road_state.ego_right,
            frame_transform=frame_transform,
            road_state_valid=road_state.has_valid_corridor,
            left_corridor_3d=road_state.left_corridor_3d,
            right_corridor_3d=road_state.right_corridor_3d,
        )
        bev_view = draw_bev_cipo(
            visual_lanes,
            processed_objs,
            cipo_status=cipo_status,
            left_corridor_3d=road_state.left_corridor_3d,
            right_corridor_3d=road_state.right_corridor_3d,
        )

        split_canvas = create_split_window(front_view, bev_view, cipo_obj, fps, canvas_size=(720, 1080))

        if writer is not None:
            writer.write(split_canvas)

        if show_gui:
            try:
                cv2.imshow(window_name, split_canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): # ESC or q
                    break
            except Exception:
                pass

        if frame_idx % 20 == 0:
            avg_fps = np.mean(fps_history[-20:])
            cipo_str = f"{cipo_obj['Z_3d']:.1f}m [{cipo_obj['status']}]" if cipo_obj else cipo_status
            print(f" Frame [{frame_idx}/{total_frames}] | Speed: {avg_fps:.1f} FPS ({frame_time*1000:.1f}ms) | CIPO: {cipo_str}")

    cap.release()
    if writer is not None:
        writer.release()
    if show_gui:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    total_time = time.time() - start_total_time
    avg_fps_overall = np.mean(fps_history) if len(fps_history) > 0 else 0.0

    print("\n================================================================")
    print(" 📊 PIPELINE PERFORMANCE & INFERENCE SUMMARY")
    print("================================================================")
    print(f" Total Frames Processed: {frame_idx}")
    print(f" Total Elapsed Time:     {total_time:.2f} seconds")
    print(f" Average Overall Speed:  {avg_fps_overall:.2f} FPS")
    print(f" Peak Frame Speed:       {np.max(fps_history):.2f} FPS")
    print(f" Annotated Output Video: {output_path}")
    print("================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time 3D Lane + Drivable Area + YOLO CIPO ADAS Pipeline")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to input MP4 video")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to output MP4 video")
    parser.add_argument("--no-gui", action="store_true", help="Disable live GUI window display")
    parser.add_argument("--no-drivable", action="store_true", help="Disable drivable area corridor overlay")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of processed frames for benchmarking")
    args = parser.parse_args()

    run_cipo_pipeline(
        video_path=args.video,
        output_path=args.output,
        show_gui=not args.no_gui,
        max_frames=args.max_frames,
        show_drivable=not args.no_drivable
    )
