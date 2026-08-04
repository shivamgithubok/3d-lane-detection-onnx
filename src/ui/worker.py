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
from src.utils.drivable_area import extract_ego_corridor_3d
from src.tracking.lane_association import LaneTrackerManager
from src.inference.cipo_tracker import CIPOTracker, DEFAULT_P_MATRIX
from src.inference.trt_yolo_detector import TRTYOLOVehicleDetector
from src.inference.trt_depth_estimator import TRTMonocularDepthEstimator
from src.inference.object_detector import OfflineYOLOVehicleDetector
from src.utils.split_visualization import draw_front_view_cipo

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    img = np.ascontiguousarray(img)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    mask = np.ascontiguousarray(mask)
    return img, mask, resized

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
        # Initialize YOLO detector once with ByteTrack persistence
        self.detector = OfflineYOLOVehicleDetector()

    def run(self):
        self.running = True
        use_trt = False

        cuda_ctx = None
        trt_engine = None
        trt_context = None

        depth_estimator = None
        tracker = None
        tracker_manager = LaneTrackerManager(max_missed_frames=10, dist_threshold=2.5)

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

                # 2. YOLO Vehicle Detector (ByteTrack) already initialized as self.detector
                detector = self.detector

                # 3. Load Monocular Depth Estimator Engine
                if os.path.exists(self.depth_engine_path):
                    depth_estimator = TRTMonocularDepthEstimator(self.depth_engine_path)

                # 4. Initialize CIPO Tracker
                tracker = CIPOTracker(P_matrix=DEFAULT_P_MATRIX, danger_dist=15.0, warning_dist=30.0)

                use_trt = True
                self.status_message.emit("🚀 Triple-TensorRT Engines Ready (Lanes + YOLO + MiDaS Depth)")
            except Exception as e:
                self.status_message.emit(f"TensorRT Init Warning: {e}")

        cap = None
        if self.video_path and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            self.status_message.emit(f"Playing Video: {os.path.basename(self.video_path)}")

        fps_history = []

        try:
            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                t0 = time.perf_counter()

                frame = None
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                proposals = []
                processed_objs = []
                cipo_obj = None
                cipo_status = "SAFE"

                if frame is not None and use_trt:
                    # Step A: 3D Lane TensorRT Inference
                    img_tensor, mask_tensor, _ = preprocess_frame(frame)
                    cuda.memcpy_htod_async(d_img, img_tensor, stream)
                    cuda.memcpy_htod_async(d_mask, mask_tensor, stream)

                    trt_context.execute_async_v3(stream.handle)

                    cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
                    cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
                    stream.synchronize()

                    raw_proposals, scores = postprocess_onnx_output(h_reg_proposals)
                    smoothed_proposals = tracker_manager.update(raw_proposals, dt=0.033)
                    proposals = smoothed_proposals if len(smoothed_proposals) > 0 else raw_proposals

                    # Step B: YOLO Vehicle Detection & Depth Map Estimation
                    raw_detections = detector.detect(frame) if detector else []
                    depth_map = None
                    if depth_estimator:
                        depth_map, _, _ = depth_estimator.estimate_depth_map(frame)

                    # Step C: CIPO Tracker & 3D In-Path Association
                    h_frame, w_frame = frame.shape[:2]
                    if tracker and (raw_detections or proposals is not None):
                        processed_objs, _ = tracker.process_detections(
                            raw_detections,
                            proposals,
                            frame_size=(w_frame, h_frame),
                            depth_map=depth_map,
                            depth_estimator=depth_estimator
                        )

                        # Find Closest In-Path Object (CIPO)
                        in_path_objs = [obj for obj in processed_objs if obj['in_path']]
                        if in_path_objs:
                            cipo_obj = min(in_path_objs, key=lambda o: o['Z_3d'])
                            dist_z = cipo_obj['Z_3d']
                            cipo_status = "DANGER" if dist_z < 15.0 else ("WARNING" if dist_z < 30.0 else "SAFE")
                            cipo_obj['status'] = cipo_status

                # Step D: Extract Ego Corridor 3D
                left_3d, right_3d = extract_ego_corridor_3d(proposals, anchor_len=20) if proposals is not None else (None, None)

                # Step E: Render Front View Overlay (Lanes + Drivable Corridor + 3D Bboxes)
                if frame is not None:
                    annotated_frame = draw_front_view_cipo(
                        frame, proposals, processed_objs, cipo_obj, DEFAULT_P_MATRIX, show_drivable=True
                    )
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)

                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                fps = 1.0 / max(0.001, (t1 - t0))
                fps_history.append(fps)
                avg_fps = float(np.mean(fps_history[-30:]))

                # Emit signal to GUI
                self.frame_processed.emit(
                    frame_rgb, proposals, processed_objs, cipo_obj, cipo_status, left_3d, right_3d, avg_fps, latency_ms
                )

                sleep_ms = max(1, int(33 - latency_ms))
                self.msleep(sleep_ms)

        finally:
            if cap:
                cap.release()
            if cuda_ctx:
                cuda_ctx.pop()

    def stop(self):
        self.running = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused
