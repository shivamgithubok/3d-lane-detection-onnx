import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi/2 + cam_pitch), -np.sin(np.pi/2 + cam_pitch), cam_height],
        [0, np.sin(np.pi/2 + cam_pitch),  np.cos(np.pi/2 + cam_pitch), 0]
    ])
    return K @ P_g2c

def homography_crop_resize(org_hw, crop_y, resize_hw):
    ratio_x = resize_hw[1] / org_hw[1]
    ratio_y = resize_hw[0] / (org_hw[0] - crop_y)
    return np.array([[ratio_x, 0, 0],
                     [0, ratio_y, -ratio_y * crop_y],
                     [0, 0, 1]])

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480
K = np.array([[2015.0, 0.0, 960.0], [0.0, 2015.0, 540.0], [0.0, 0.0, 1.0]])
H_crop = homography_crop_resize([1280, 1920], 0, [360, 480])
CAM_HEIGHT = 1.5

cap = cv2.VideoCapture("videos/input_0.mp4")
ret, frame = cap.read()
resized = cv2.resize(frame, (INPUT_W, INPUT_H))
img = resized[:, :, ::-1].astype(np.float32)
img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)

sess = ort.InferenceSession("models/anchor3dlane_raw.onnx")
reg_proposals, anchors = sess.run(None, {'img': img, 'mask': mask})
proposals, scores = postprocess_onnx_output(reg_proposals)

# try a spread of pitch values
for pitch_deg in [-2,-3, 6, 9, 12, 15, 18]:
    P_g2im = projection_g2im(np.radians(pitch_deg), CAM_HEIGHT, K)
    P_final = H_crop @ P_g2im

    canvas = resized.copy()
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_final)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < INPUT_W and 0 <= v < INPUT_H]
        for i in range(1, len(draw_pts)):
            cv2.line(canvas, draw_pts[i-1], draw_pts[i], (0, 0, 255), 2)

    cv2.putText(canvas, f"pitch={pitch_deg}deg", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.imwrite(f"output/tune_pitch_{pitch_deg}.png", canvas)
    print(f"Saved output/tune_pitch_{pitch_deg}.png")