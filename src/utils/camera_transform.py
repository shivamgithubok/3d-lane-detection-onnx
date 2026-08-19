"""One source of truth for source-image ↔ model-image geometry.

The deployed lane engine was validated with the complete Garmin frame resized
to 480x360.  A centre crop looks geometrically cleaner, but discards the road
boundaries that the current model actually relies on.  Keep the full FOV by
default and retain the exact inverse affine mapping for overlays and object
ground-contact points.  Any crop must be enabled only after model validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraTransform:
    """Geometry of an aspect-preserving source → model image conversion."""

    source_width: int
    source_height: int
    model_width: int = 480
    model_height: int = 360
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: float = 0.0
    crop_height: float = 0.0

    @classmethod
    def for_frame(
        cls,
        frame: np.ndarray,
        model_size: tuple[int, int] = (480, 360),
        sky_crop_px: int = 0,
    ) -> "CameraTransform":
        h, w = frame.shape[:2]
        model_w, model_h = model_size
        if h <= 0 or w <= 0:
            raise ValueError("Cannot build a camera transform for an empty frame")

        # Do not crop left/right: it removed ~145 px from both sides of the
        # 960x502 Garmin input and materially reduced lane recall.  Sky crop
        # remains opt-in for a future, validated preprocessing sweep.
        sky_crop_px = max(0, min(int(sky_crop_px), h - 2))
        return cls(
            w, h, model_w, model_h,
            crop_x=0.0,
            crop_y=float(sky_crop_px),
            crop_width=float(w),
            crop_height=float(h - sky_crop_px),
        )

    @property
    def scale_x(self) -> float:
        return float(self.model_width) / self.crop_width

    @property
    def scale_y(self) -> float:
        return float(self.model_height) / self.crop_height

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Crop and resize a BGR source frame into the model's native geometry."""
        x0 = max(0, int(np.floor(self.crop_x)))
        y0 = max(0, int(np.floor(self.crop_y)))
        x1 = min(self.source_width, int(np.ceil(self.crop_x + self.crop_width)))
        y1 = min(self.source_height, int(np.ceil(self.crop_y + self.crop_height)))
        cropped = frame[y0:y1, x0:x1]
        if cropped.size == 0:
            raise ValueError("Camera crop is empty")
        return cv2.resize(cropped, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)

    def source_to_model(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = pts.copy()
        out[:, 0] = (pts[:, 0] - self.crop_x) * self.scale_x
        out[:, 1] = (pts[:, 1] - self.crop_y) * self.scale_y
        return out

    def model_to_source(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = pts.copy()
        out[:, 0] = pts[:, 0] / self.scale_x + self.crop_x
        out[:, 1] = pts[:, 1] / self.scale_y + self.crop_y
        return out

    def source_point_is_visible_to_model(self, u: float, v: float) -> bool:
        return (
            self.crop_x <= u <= self.crop_x + self.crop_width
            and self.crop_y <= v <= self.crop_y + self.crop_height
        )
