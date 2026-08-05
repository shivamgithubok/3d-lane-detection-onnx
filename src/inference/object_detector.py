import cv2
import numpy as np
import os

class OfflineYOLOVehicleDetector:
    """
    High-performance YOLO Vehicle Tracker using ByteTrack and PyTorch model weights (yolov8n.pt).
    Strictly detects and tracks Cars (COCO 2) and Trucks (COCO 7).
    """
    def __init__(self, model_path="models/yolov8n.pt", conf_thresh=0.25):
        self.conf_thresh = conf_thresh
        self.model = None

        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path, task='detect')
                print(f"[ObjectDetector] Successfully loaded YOLO ByteTrack model from {model_path}")
            except Exception as e:
                print(f"[ObjectDetector] Ultralytics loading failed: {e}")

    def detect(self, frame):
        if self.model is not None:
            try:
                # ByteTrack multi-object tracking for Cars (class 2) and Trucks (class 7)
                results = self.model.track(frame, persist=True, imgsz=(384, 480), tracker="bytetrack.yaml", classes=[2, 7], verbose=False, conf=self.conf_thresh)[0]
                detections = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = 'car' if cls_id == 2 else 'truck'
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    detections.append({
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        'conf': conf,
                        'class': label,
                        'track_id': track_id
                    })
                return detections
            except Exception as e:
                print(f"[ObjectDetector] Detection error: {e}")

        return []
