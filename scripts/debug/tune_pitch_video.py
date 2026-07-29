import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels

ENGINE_PATH = "models/anchor3dlane_raw.engine"
VIDEO_PATH = "data/images/example_2.mp4"

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

def projection_g2im(cam_pitch_rad, cam_height, K):
    P_g2c = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi/2 + cam_pitch_rad), -np.sin(np.pi/2 + cam_pitch_rad), cam_height],
        [0, np.sin(np.pi/2 + cam_pitch_rad),  np.cos(np.pi/2 + cam_pitch_rad), 0]
    ])
    return K @ P_g2c

def homography_crop_resize(org_hw, crop_y, resize_hw):
    ratio_x = resize_hw[1] / org_hw[1]
    ratio_y = resize_hw[0] / (org_hw[0] - crop_y)
    return np.array([[ratio_x, 0, 0],
                     [0, ratio_y, -ratio_y * crop_y],
                     [0, 0, 1]])

def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    return img, mask, resized

def main():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

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

    context.set_tensor_address("img", int(d_img))
    context.set_tensor_address("mask", int(d_mask))
    context.set_tensor_address("reg_proposals", int(d_reg_proposals))
    context.set_tensor_address("anchors", int(d_anchors))

    cap = cv2.VideoCapture(VIDEO_PATH)
    for _ in range(60): # skip 60 frames to get a good road view
        cap.read()
    ret, frame = cap.read()
    cap.release()

    img, mask, resized = preprocess(frame)
    h_img = np.ascontiguousarray(img)
    h_mask = np.ascontiguousarray(mask)

    cuda.memcpy_htod_async(d_img, h_img, stream)
    cuda.memcpy_htod_async(d_mask, h_mask, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
    cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
    stream.synchronize()

    proposals, scores = postprocess_onnx_output(h_reg_proposals, conf_threshold=0.15)

    K = np.array([[2015.0, 0.0, 960.0], [0.0, 2015.0, 540.0], [0.0, 0.0, 1.0]])
    H_crop = homography_crop_resize([1280, 1920], 0, [360, 480])

    os.makedirs("output/calibration_tune", exist_ok=True)

    # Test matrix 1: OpenLane / Waymo standard matrix (infer_image.py matrix)
    P_openlane = np.array([
        [517.368023818057, -1.2192886414581698, 245.2892094158112, -1317.8869498335648],
        [-2.080056338959641, 584.7261604964109, 187.72380574460553, -379.84979423293373],
        [-0.012627197281367759, -0.004025390168358267, 0.9999121712044562, -2.096266740239456]
    ])

    canvas_openlane = resized.copy()
    for lane in proposals:
        pts = decode_lane_pixels(lane, P_openlane)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < INPUT_W and 0 <= v < INPUT_H]
        for i in range(1, len(draw_pts)):
            cv2.line(canvas_openlane, draw_pts[i-1], draw_pts[i], (0, 255, 0), 2)
    cv2.imwrite("output/calibration_tune/matrix_openlane.jpg", canvas_openlane)

    # Test pitch variations: pitch_deg from -5 to +15
    for pitch_deg in [-5, -2, 0, 2, 4, 6, 8, 10, 12, 15]:
        for cam_h in [1.5, 1.7]:
            P_g2im = projection_g2im(np.radians(pitch_deg), cam_h, K)
            P_final = H_crop @ P_g2im

            canvas = resized.copy()
            for lane in proposals:
                pts = decode_lane_pixels(lane, P_final)
                draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < INPUT_W and 0 <= v < INPUT_H]
                for i in range(1, len(draw_pts)):
                    cv2.line(canvas, draw_pts[i-1], draw_pts[i], (0, 255, 0), 2)

            cv2.putText(canvas, f"pitch={pitch_deg}deg h={cam_h}m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(f"output/calibration_tune/pitch_{pitch_deg}_h_{cam_h}.jpg", canvas)

    print("Saved tuning frames in output/calibration_tune/")

if __name__ == "__main__":
    main()
