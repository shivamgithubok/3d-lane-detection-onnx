import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels

MODEL_PATH = "models/anchor3dlane_raw.onnx"
IMAGE_PATH = "data/images/video_frame_60.jpg"
OUTPUT_PATH = "output/video_frame_60_annotated.jpg"

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

P_MATRIX = np.array([
    [517.368023818057, -1.2192886414581698, 245.2892094158112, -1317.8869498335648],
    [-2.080056338959641, 584.7261604964109, 187.72380574460553, -379.84979423293373],
    [-0.012627197281367759, -0.004025390168358267, 0.9999121712044562, -2.096266740239456]
])

def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized[:, :, ::-1].astype(np.float32)          # BGR -> RGB
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)  # confirmed correct convention
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
    print("Loading ONNX model...")
    providers = ort.get_available_providers()
    print("Available execution providers:", providers)
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)

    print(f"Reading image from {IMAGE_PATH}...")
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Failed to read image {IMAGE_PATH}")
        return

    img, mask, resized = preprocess(frame)

    t0 = time.time()
    reg_proposals, anchors = sess.run(None, {'img': img, 'mask': mask})
    proposals, scores = postprocess_onnx_output(reg_proposals)
    t1 = time.time()
    
    print(f"Inference time: {t1 - t0:.4f}s")
    num_lanes = 0 if proposals is None else len(proposals)
    print(f"Found {num_lanes} lanes.")

    annotated = draw_lanes(resized.copy(), proposals)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
