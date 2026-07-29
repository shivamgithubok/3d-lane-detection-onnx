import time
import numpy as np
import onnxruntime as ort
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.postprocess import postprocess_onnx_output

MODEL_PATH = "models/anchor3dlane_raw.onnx"
INPUT_H, INPUT_W = 360, 480
WARMUP_RUNS = 15
TEST_RUNS = 100

def benchmark():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        return

    print("Checking ONNX Runtime Execution Providers...")
    available_providers = ort.get_available_providers()
    print(f"Available Providers: {available_providers}")

    preferred_providers = [
        ('TensorRTExecutionProvider', {'trt_fp16_enable': True}),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    
    # Filter to matching available providers
    active_providers = []
    for p in preferred_providers:
        p_name = p[0] if isinstance(p, tuple) else p
        if p_name in available_providers:
            active_providers.append(p)
            
    if not active_providers:
        active_providers = available_providers

    print(f"Initializing Session with Providers: {active_providers}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess = ort.InferenceSession(MODEL_PATH, sess_options=session_options, providers=active_providers)
    
    actual_provider = sess.get_providers()
    print(f"Active Provider(s) in Session: {actual_provider}")

    # Generate synthetic input tensors matching model contract
    IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    IMG_NORM_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    dummy_frame = np.random.randint(0, 256, (INPUT_H, INPUT_W, 3), dtype=np.uint8)
    img = (dummy_frame[:, :, ::-1].astype(np.float32) - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
    
    feed = {'img': img, 'mask': mask}

    print(f"\nWarming up model for {WARMUP_RUNS} iterations...")
    for _ in range(WARMUP_RUNS):
        sess.run(None, feed)

    print(f"Benchmarking inference performance over {TEST_RUNS} iterations...")
    inf_times = []
    post_times = []

    for _ in range(TEST_RUNS):
        t0 = time.perf_counter()
        reg_proposals, anchors = sess.run(None, feed)
        t1 = time.perf_counter()
        
        proposals, scores = postprocess_onnx_output(reg_proposals)
        t2 = time.perf_counter()

        inf_times.append((t1 - t0) * 1000.0)    # in ms
        post_times.append((t2 - t1) * 1000.0)   # in ms

    avg_inf_ms = np.mean(inf_times)
    std_inf_ms = np.std(inf_times)
    avg_post_ms = np.mean(post_times)
    total_ms = avg_inf_ms + avg_post_ms
    pure_fps = 1000.0 / avg_inf_ms
    e2e_fps = 1000.0 / total_ms

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Active Provider       : {actual_provider[0]}")
    print(f"Model Inference Time  : {avg_inf_ms:.2f} ms ± {std_inf_ms:.2f} ms")
    print(f"Pure Model FPS        : {pure_fps:.2f} FPS")
    print(f"Postprocessing Time   : {avg_post_ms:.2f} ms")
    print(f"Total Pipeline Latency: {total_ms:.2f} ms")
    print(f"End-to-End Pipeline   : {e2e_fps:.2f} FPS")
    print("====================================================\n")

if __name__ == "__main__":
    benchmark()
