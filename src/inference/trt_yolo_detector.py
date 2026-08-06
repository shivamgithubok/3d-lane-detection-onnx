import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

class TRTYOLOVehicleDetector:
    def __init__(self, engine_path="models/yolov8n.engine", conf_thresh=0.25, iou_thresh=0.45):
        self.engine_path = engine_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_shape = (1, 3, 384, 480)
        self.output_shape = (1, 84, 3780)

        # Load TensorRT Engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate CUDA memory
        self.h_input = np.empty(self.input_shape, dtype=np.float32)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Set TensorRT execution tensor addresses
        self.context.set_tensor_address("images", int(self.d_input))
        self.context.set_tensor_address("output0", int(self.d_output))

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (480, 384))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(norm), w, h

    def postprocess(self, output, orig_w, orig_h):
        # output shape: (1, 84, 3780) -> transpose to (3780, 84)
        predictions = output[0].T
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        # Filter for vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        vehicle_indices = list(VEHICLE_CLASSES.keys())
        vehicle_scores = scores[:, vehicle_indices]

        class_ids = np.argmax(vehicle_scores, axis=1)
        confidences = np.max(vehicle_scores, axis=1)

        mask = confidences > self.conf_thresh
        if not np.any(mask):
            return []

        valid_boxes = boxes[mask]
        valid_confs = confidences[mask]
        valid_class_ids = [vehicle_indices[idx] for idx in class_ids[mask]]

        scale_x = orig_w / 480.0
        scale_y = orig_h / 384.0

        boxes_xywh = []
        for box in valid_boxes:
            cx, cy, w, h = box
            x1 = (cx - w / 2.0) * scale_x
            y1 = (cy - h / 2.0) * scale_y
            bw = w * scale_x
            bh = h * scale_y
            boxes_xywh.append([int(x1), int(y1), int(bw), int(bh)])

        indices = cv2.dnn.NMSBoxes(boxes_xywh, valid_confs.tolist(), self.conf_thresh, self.iou_thresh)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes_xywh[i]
                x2 = x1 + w
                y2 = y1 + h
                cls_id = valid_class_ids[i]
                label = VEHICLE_CLASSES.get(cls_id, 'car')
                
                detections.append({
                    'bbox': [max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)],
                    'conf': float(valid_confs[i]),
                    'class': label,
                    'track_id': i + 1
                })

        return detections

    def detect(self, frame):
        input_tensor, orig_w, orig_h = self.preprocess(frame)
        cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.postprocess(self.h_output, orig_w, orig_h)
