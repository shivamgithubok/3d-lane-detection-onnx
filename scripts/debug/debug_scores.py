import cv2
import numpy as np
import onnxruntime as ort

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480
ANCHOR_LEN = 20

def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# change this to the frame number from the dense-traffic screenshot
TARGET_FRAME = 200

cap = cv2.VideoCapture("videos/input.mp4")
for _ in range(TARGET_FRAME):
    cap.read()
ret, frame = cap.read()

resized = cv2.resize(frame, (INPUT_W, INPUT_H))
img = resized[:, :, ::-1].astype(np.float32)
img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)

sess = ort.InferenceSession("models/anchor3dlane_raw.onnx")
reg_proposals, anchors = sess.run(None, {'img': img, 'mask': mask})
proposals = reg_proposals[0]
logits = softmax(proposals[:, 5 + 3*ANCHOR_LEN:], axis=1)
score = 1 - logits[:, 0]

print("Top 10 raw scores (before any threshold):", sorted(score, reverse=True)[:10])