import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels, ANCHOR_Y_STEPS
from src.utils.visualization import draw_bev

MODEL_PATH = "models/anchor3dlane_raw.onnx"
VIDEO_PATH = "videos/input_0.mp4"

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

# P_MATRIX = np.array([
#     [517.368023818057, -1.2192886414581698, 245.2892094158112, -1317.8869498335648],
#     [-2.080056338959641, 584.7261604964109, 187.72380574460553, -379.84979423293373],
#     [-0.012627197281367759, -0.004025390168358267, 0.9999121712044562, -2.096266740239456]
# ])
P_MATRIX = np.array([[503.75, 239.853798484583, 8.37587920860026, 0.0], 
                     [0.0, 171.5606810003957, -561.0731591880358, 850.078125], 
                     [0.0, 0.9993908270190958, 0.03489949670250108, 0.0]])


def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    return img, mask, resized

def draw_lanes(frame, proposals):
    if proposals is None:
        return frame
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_MATRIX)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]]
        for i in range(1, len(draw_pts)):
            cv2.line(frame, draw_pts[i-1], draw_pts[i], (0, 0, 255), 2)
    return frame

def main():
    providers = ort.get_available_providers()
    print("Available execution providers:", providers)
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    cap = cv2.VideoCapture(VIDEO_PATH)

    cv2.namedWindow("Front View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV (raw model output, no calibration)", cv2.WINDOW_NORMAL)

    frame_count, total_time = 0, 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img, mask, resized = preprocess(frame)

        t0 = time.time()
        reg_proposals, anchors = sess.run(None, {'img': img, 'mask': mask})
        proposals, scores = postprocess_onnx_output(reg_proposals)
        t1 = time.time()
        total_time += (t1 - t0)
        frame_count += 1

        front_annotated = draw_lanes(resized.copy(), proposals)
        bev = draw_bev(proposals, ANCHOR_Y_STEPS)

        cv2.imshow("Front View", cv2.resize(front_annotated, (960, 720)))
        cv2.imshow("BEV (raw model output, no calibration)", bev)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Avg FPS: {frame_count/total_time:.2f}")

if __name__ == "__main__":
    main()
    