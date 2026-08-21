"""
PySide6 Async Triple-TensorRT Full ADAS Pipeline Worker Thread
Runs:
 1. 3D Anchor Lane TensorRT Engine
 2. YOLOv8 Vehicle Detector TensorRT Engine
 3. MiDaS Monocular Depth Estimator TensorRT Engine
 4. CIPO Tracker & Drivable Area Corridor Pipeline
"""

import cv2
import numpy as np
import os
import time
from PySide6.QtCore import QThread, Signal
from src.inference.postprocess import postprocess_onnx_output
from src.inference.lane_preprocess import prepare_lane_input
from src.inference.cipo_tracker import CIPOTracker
from src.inference.trt_depth_estimator import TRTMonocularDepthEstimator
from src.inference.object_detector import OfflineYOLOVehicleDetector
from src.utils.split_visualization import draw_front_view_cipo
from src.utils.camera_transform import CameraTransform
from src.tracking.road_state import RoadStateEstimator
from src.utils.ego_speed import EgoSpeedLog
from src.utils.calibration import (
    make_P_matrix,
    preset_for_video,
    OPENLANE_CAM_PITCH_DEG,
    OPENLANE_CAM_HEIGHT,
)

INPUT_H, INPUT_W = 360, 480

def preprocess_frame(frame):
    frame_transform = CameraTransform.for_frame(frame, (INPUT_W, INPUT_H))
    resized = frame_transform.apply(frame)
    img, mask, meta = prepare_lane_input(resized)
    return img, mask, resized, frame_transform, meta

class InferenceWorker(QThread):
    # Signal emitted to UI: frame_rgb, proposals, processed_objs, cipo_obj, cipo_status, left_3d, right_3d, avg_fps, latency_ms
    frame_processed = Signal(np.ndarray, list, list, object, str, object, object, float, float)
    status_message = Signal(str)

    def __init__(self, video_path=None, model_path="models/anchor3dlane_raw.engine", parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.engine_path = model_path if model_path.endswith('.engine') else "models/anchor3dlane_raw.engine"
        self.yolo_engine_path = "models/yolov8n.engine"
        self.depth_engine_path = "models/monocular_depth.engine"
        self.running = False
        self.paused = False
        # YOLO is created on the worker thread after CUDA is ready (avoids empty/missed frames)
        self.detector = None
        # P is LOCKED to OpenLane training extrinsics. Retuning pitch (e.g. Garmin -6°)
        # shears the front corridor vs cyan lanes — model 3D assumes this camera.
        pitch, height = preset_for_video(video_path)
        self.calib_pitch = float(pitch)
        self.calib_height = float(height)
        self.P_matrix = make_P_matrix(OPENLANE_CAM_PITCH_DEG, OPENLANE_CAM_HEIGHT)

    def set_calibration(self, pitch_deg, height_m):
        """
        Cal panel no longer rebuilds P. Live pitch/height changes break
        model-3D ↔ image projection consistency (narrow/skewed red corridor).
        """
        self.calib_pitch = OPENLANE_CAM_PITCH_DEG
        self.calib_height = OPENLANE_CAM_HEIGHT
        self.P_matrix = make_P_matrix(OPENLANE_CAM_PITCH_DEG, OPENLANE_CAM_HEIGHT)

    def run(self):
        self.running = True
        use_trt = False

        cuda_ctx = None
        trt_engine = None
        trt_context = None

        depth_estimator = None
        tracker = None
        detector = None
        road_state_estimator = RoadStateEstimator()
        speed_log = EgoSpeedLog.auto_load(self.video_path) if self.video_path else None
        if speed_log is not None:
            self.status_message.emit(f"Loaded ego speed JSON ({len(speed_log.mps)} frames)")

        # Initialize CUDA Context & Triple TensorRT Engines
        if os.path.exists(self.engine_path):
            try:
                import tensorrt as trt
                import pycuda.driver as cuda

                cuda.init()
                dev = cuda.Device(0)
                cuda_ctx = dev.make_context()

                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                
                # 1. Load 3D Lane TensorRT Engine
                self.status_message.emit("Loading 3D Lane TensorRT Engine...")
                with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    trt_engine = runtime.deserialize_cuda_engine(f.read())
                trt_context = trt_engine.create_execution_context()

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

                trt_context.set_tensor_address("img", int(d_img))
                trt_context.set_tensor_address("mask", int(d_mask))
                trt_context.set_tensor_address("reg_proposals", int(d_reg_proposals))
                trt_context.set_tensor_address("anchors", int(d_anchors))

                # 2. YOLO nano + ByteTrack (pop pycuda so Ultralytics/TensorRT can load)
                detector = None
                try:
                    cuda_ctx.pop()
                except Exception:
                    pass
                try:
                    yolo_path = self.yolo_engine_path
                    if not os.path.isfile(yolo_path):
                        yolo_path = "models/yolov8n.pt"
                    self.detector = OfflineYOLOVehicleDetector(
                        model_path=yolo_path, conf_thresh=0.22, imgsz=640
                    )
                    detector = self.detector
                except Exception as ye:
                    self.status_message.emit(f"YOLO init warning: {ye}")
                    self.detector = None
                    detector = None
                finally:
                    try:
                        cuda_ctx.push()
                    except Exception:
                        pass

                # 3. Load Monocular Depth Estimator Engine
                if os.path.exists(self.depth_engine_path):
                    depth_estimator = TRTMonocularDepthEstimator(self.depth_engine_path)

                # 4. Initialize CIPO Tracker
                tracker = CIPOTracker(P_matrix=self.P_matrix, danger_dist=15.0, warning_dist=30.0)
                self.status_message.emit(
                    f"Calib pitch={self.calib_pitch:.1f}° h={self.calib_height:.1f}m"
                )

                use_trt = True
                self.status_message.emit("🚀 Triple-TensorRT Engines Ready (Lanes + YOLO + MiDaS Depth)")
            except Exception as e:
                self.status_message.emit(f"TensorRT Init Warning: {e}")

        cap = None
        if self.video_path and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            self.status_message.emit(f"Playing Video: {os.path.basename(self.video_path)}")

        fps_history = []
        frame_i = 0
        last_depth_map = None
        last_source_ms = None
        DEPTH_EVERY_N = 2  # reuse depth on alternate frames → big FPS win on Orin

        try:
            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                t0 = time.perf_counter()

                frame = None
                frame_transform = None
                source_dt = 1.0 / 30.0
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        last_source_ms = None
                        continue
                    source_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if last_source_ms is not None and source_ms > last_source_ms:
                        source_dt = (source_ms - last_source_ms) / 1000.0
                    last_source_ms = source_ms

                proposals = []
                processed_objs = []
                cipo_obj = None
                cipo_status = "SAFE"
                ego_left, ego_right = None, None

                if frame is not None and use_trt:
                    # Step A: 3D Lane TensorRT Inference
                    img_tensor, mask_tensor, _, frame_transform, prep_meta = preprocess_frame(frame)
                    cuda.memcpy_htod_async(d_img, img_tensor, stream)
                    cuda.memcpy_htod_async(d_mask, mask_tensor, stream)

                    trt_context.execute_async_v3(stream.handle)

                    cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
                    cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
                    stream.synchronize()

                    raw_proposals, scores = postprocess_onnx_output(
                        h_reg_proposals, conf_threshold=prep_meta["conf"]
                    )
                    speed_mps = speed_log.get_mps(frame_i) if speed_log is not None else None
                    road_state = road_state_estimator.update(
                        raw_proposals, dt=source_dt, speed_mps=speed_mps
                    )
                    # Render immediate measurements while tracks acquire; only
                    # confirmed/predicted tracks may feed CIPO safety logic.
                    proposals = road_state.visual_lanes
                    safety_lanes = road_state.lanes

                    # Step B: YOLO Vehicle Detection & Depth Map Estimation
                    # Pop pycuda context so Ultralytics/TensorRT YOLO can use the GPU (P0)
                    if cuda_ctx is not None:
                        try:
                            cuda_ctx.pop()
                        except Exception:
                            pass
                    try:
                        raw_detections = detector.detect(frame) if detector else []
                    finally:
                        if cuda_ctx is not None:
                            try:
                                cuda_ctx.push()
                            except Exception:
                                pass
                    depth_map = last_depth_map
                    if depth_estimator and (frame_i % DEPTH_EVERY_N == 0 or last_depth_map is None):
                        depth_map, _, _ = depth_estimator.estimate_depth_map(frame)
                        last_depth_map = depth_map

                    # Step C: CIPO Tracker & 3D In-Path Association
                    h_frame, w_frame = frame.shape[:2]
                    ego_left, ego_right = road_state.ego_left, road_state.ego_right
                    # Live Cal panel → keep CIPO P in sync
                    if tracker is not None:
                        tracker.P = np.asarray(self.P_matrix, dtype=np.float64)
                    if tracker and (raw_detections or proposals is not None):
                        processed_objs, cipo_obj = tracker.process_detections(
                            raw_detections,
                            safety_lanes,
                            frame_size=(w_frame, h_frame),
                            depth_map=depth_map,
                            depth_estimator=depth_estimator,
                            ego_left=ego_left,
                            ego_right=ego_right,
                            frame_transform=frame_transform,
                            road_state_confirmed=road_state.is_confirmed,
                        )

                        # Find Closest In-Path Object (CIPO)
                        in_path_objs = [obj for obj in processed_objs if obj['in_path']]
                        if in_path_objs and road_state.is_confirmed:
                            cipo_obj = min(in_path_objs, key=lambda o: o['Z_3d'])
                            dist_z = cipo_obj['Z_3d']
                            cipo_status = "DANGER" if dist_z < 15.0 else ("WARNING" if dist_z < 30.0 else "SAFE")
                            cipo_obj['status'] = cipo_status
                        elif road_state.status != "CONFIRMED":
                            cipo_status = "DEGRADED"

                # Step D: All rendering reads the same validated temporal road state.
                if frame is not None and use_trt:
                    left_3d = road_state.left_corridor_3d
                    right_3d = road_state.right_corridor_3d
                else:
                    left_3d, right_3d = None, None

                # Step E: Render Front View Overlay (Lanes + Drivable Corridor + 3D Bboxes)
                if frame is not None:
                    annotated_frame = draw_front_view_cipo(
                        frame,
                        proposals,
                        processed_objs,
                        cipo_obj,
                        np.asarray(self.P_matrix, dtype=np.float64),
                        show_drivable=True,
                        ego_left=ego_left,
                        ego_right=ego_right,
                        frame_transform=frame_transform,
                        road_state_valid=(left_3d is not None and right_3d is not None),
                        left_corridor_3d=left_3d,
                        right_corridor_3d=right_3d,
                    )
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    # Downscale for UI transfer/paint (keeps HUD readable, cuts Qt cost)
                    max_w = 960
                    if frame_rgb.shape[1] > max_w:
                        scale = max_w / float(frame_rgb.shape[1])
                        frame_rgb = cv2.resize(
                            frame_rgb,
                            (max_w, int(frame_rgb.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                else:
                    frame_rgb = np.zeros((405, 720, 3), dtype=np.uint8)

                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                fps = 1.0 / max(0.001, (t1 - t0))
                fps_history.append(fps)
                avg_fps = float(np.mean(fps_history[-30:]))

                # Emit signal to GUI
                self.frame_processed.emit(
                    frame_rgb, proposals, processed_objs, cipo_obj, cipo_status, left_3d, right_3d, avg_fps, latency_ms
                )

                frame_i += 1
                # Target ~15 FPS pacing when pipeline is fast enough
                sleep_ms = max(1, int(66 - latency_ms))
                self.msleep(sleep_ms)

        finally:
            if cap:
                cap.release()
            
            # Explicitly release CUDA memory buffers & engine contexts before popping context
            try:
                if 'trt_context' in locals() and trt_context is not None:
                    del trt_context
                if 'trt_engine' in locals() and trt_engine is not None:
                    del trt_engine
                if 'stream' in locals() and stream is not None:
                    del stream
                if 'd_img' in locals() and d_img is not None:
                    del d_img
                if 'd_mask' in locals() and d_mask is not None:
                    del d_mask
                if 'd_reg_proposals' in locals() and d_reg_proposals is not None:
                    del d_reg_proposals
                if 'd_anchors' in locals() and d_anchors is not None:
                    del d_anchors
                if hasattr(self, 'detector') and self.detector is not None:
                    # Clean up the YOLO model and its associated CUDA states if possible
                    if hasattr(self.detector, 'model') and self.detector.model is not None:
                        self.detector.model = None
            except Exception as ce:
                print(f"[Worker Cleanup Warning] {ce}")

            if cuda_ctx:
                try:
                    cuda_ctx.pop()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused
