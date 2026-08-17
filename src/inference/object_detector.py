import os

import numpy as np

# COCO vehicle classes used for ADAS
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_TRACKER_YAML = os.path.join(os.path.dirname(__file__), "bytetrack_lowfps.yaml")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


class OfflineYOLOVehicleDetector:
    """
    YOLOv8-nano + ByteTrack for vehicles.

    P0: never emit a hard-empty frame on a miss/error — coast last tracks.
    P1: imgsz=640, car/moto/bus/truck, low-FPS ByteTrack, IoU ID recovery.
    """

    def __init__(
        self,
        model_path="models/yolov8n.engine",
        conf_thresh=0.22,
        imgsz=640,
        max_coast_frames=10,
    ):
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.max_coast_frames = max_coast_frames
        self.model = None
        self.model_path = None
        self._last_dets = []
        self._coast_frames = 0
        self._next_fallback_id = 10000

        self._load_model(model_path)

    def _candidate_paths(self, preferred):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        models = os.path.join(root, "models")
        names = []
        if preferred:
            names.append(preferred)
        # Prefer Orin-compiled engine, then PT (Ultralytics export/track), then ONNX
        names.extend(
            [
                os.path.join(models, "yolov8n.engine"),
                os.path.join(models, "yolov8n.pt"),
                os.path.join(models, "yolov8n.onnx"),
            ]
        )
        seen = set()
        out = []
        for p in names:
            ap = os.path.abspath(p)
            if ap in seen or not os.path.isfile(ap):
                continue
            seen.add(ap)
            out.append(ap)
        return out

    def _load_model(self, preferred):
        try:
            from ultralytics import YOLO
        except Exception as e:
            print(f"[ObjectDetector] Ultralytics import failed: {e}")
            return

        last_err = None
        for path in self._candidate_paths(preferred):
            try:
                self.model = YOLO(path, task="detect")
                self.model_path = path
                print(f"[ObjectDetector] Loaded YOLOv8n ByteTrack from {path} (imgsz={self.imgsz})")
                return
            except Exception as e:
                last_err = e
                print(f"[ObjectDetector] Failed {path}: {e}")
        if last_err is not None:
            print(f"[ObjectDetector] No YOLO weights loaded: {last_err}")

    def _parse_results(self, results):
        detections = []
        if results is None or results.boxes is None:
            return detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = VEHICLE_CLASSES.get(cls_id, "car")
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1
            detections.append(
                {
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "conf": conf,
                    "class": label,
                    "track_id": track_id,
                }
            )
        return detections

    def _recover_ids(self, detections):
        """Assign previous IDs by IoU when ByteTrack omitted id this frame."""
        if not detections:
            return detections
        prev = [d for d in self._last_dets if d.get("track_id", -1) > 0]
        used = set()
        for det in detections:
            if det["track_id"] > 0:
                used.add(det["track_id"])
                continue
            best_iou, best_id = 0.35, None
            for p in prev:
                pid = p["track_id"]
                if pid in used:
                    continue
                iou = _iou(det["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, pid
            if best_id is not None:
                det["track_id"] = best_id
                used.add(best_id)
            else:
                det["track_id"] = self._next_fallback_id
                self._next_fallback_id += 1
        return detections

    def detect(self, frame):
        if self.model is None:
            if self._last_dets and self._coast_frames < self.max_coast_frames:
                self._coast_frames += 1
                return list(self._last_dets)
            return []

        try:
            tracker = _TRACKER_YAML if os.path.isfile(_TRACKER_YAML) else "bytetrack.yaml"
            results = self.model.track(
                frame,
                persist=True,
                imgsz=self.imgsz,
                tracker=tracker,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
                conf=self.conf_thresh,
                iou=0.50,
            )[0]
            detections = self._recover_ids(self._parse_results(results))
            if detections:
                self._last_dets = detections
                self._coast_frames = 0
                return detections

            # Empty YOLO output: coast last tracks instead of wiping the scene
            if self._last_dets and self._coast_frames < self.max_coast_frames:
                self._coast_frames += 1
                return list(self._last_dets)
            return []
        except Exception as e:
            print(f"[ObjectDetector] Detection error: {e}")
            if self._last_dets and self._coast_frames < self.max_coast_frames:
                self._coast_frames += 1
                return list(self._last_dets)
            return []
