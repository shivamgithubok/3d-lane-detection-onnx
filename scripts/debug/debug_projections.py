import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels


IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480
P_MATRIX = np.array([[503.75, 239.67108834109771, -12.560629498306522, 0.0], 
    [0.0, 122.00709288879507, -573.8906050035907, 878.4140625], 
    [0.0, 0.9986295347545738, -0.05233595624294384, 0.0]])
cap = cv2.VideoCapture("videos/input.mp4")
ret, frame = cap.read()
resized = cv2.resize(frame, (INPUT_W, INPUT_H))
img = resized[:, :, ::-1].astype(np.float32)
img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)

sess = ort.InferenceSession("models/anchor3dlane_raw.onnx")
reg_proposals, anchors = sess.run(None, {'img': img, 'mask': mask})
proposals, scores = postprocess_onnx_output(reg_proposals)

print(f"Detections: {len(proposals)}")
all_u, all_v = [], []
for lane in proposals:
    pts = decode_lane_pixels(lane, P_MATRIX)
    for u, v in pts:
        all_u.append(u)
        all_v.append(v)

print(f"u (pixel x) range: {min(all_u):.1f} to {max(all_u):.1f}   (frame width is 480)")
print(f"v (pixel y) range: {min(all_v):.1f} to {max(all_v):.1f}   (frame height is 360)")