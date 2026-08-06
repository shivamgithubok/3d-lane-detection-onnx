import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTMonocularDepthEstimator:
    def __init__(self, engine_path="models/monocular_depth.engine"):
        self.engine_path = engine_path
        self.input_shape = (1, 3, 256, 256)
        self.output_shape = (1, 256, 256)

        # ImageNet normalization parameters for MiDaS
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        # Load TensorRT Engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate CUDA memory
        self.h_input = np.empty(self.input_shape, dtype=np.float32)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Set TensorRT execution tensor addresses ('0' for input, '797' for output)
        self.context.set_tensor_address("0", int(self.d_input))
        self.context.set_tensor_address("797", int(self.d_output))

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        norm = (norm - self.mean) / self.std
        return np.ascontiguousarray(norm), w, h

    def estimate_depth_map(self, frame):
        """
        Runs TensorRT inference and returns pre-trained MiDaS relative inverse depth map (256x256).
        """
        input_tensor, orig_w, orig_h = self.preprocess(frame)
        cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        inv_depth_map = self.h_output[0] # (256, 256) MiDaS relative disparity
        return inv_depth_map, orig_w, orig_h

    def query_vehicle_depth(self, inv_depth_map, bbox, orig_w, orig_h):
        """
        Queries the pre-trained MiDaS depth map inside the vehicle's 2D bounding box and converts
        relative inverse depth to metric distance Z in meters (calibrated range 5m to 80m).
        """
        x1, y1, x2, y2 = bbox
        scale_u = 256.0 / float(orig_w)
        scale_v = 256.0 / float(orig_h)

        u1 = max(0, int(x1 * scale_u))
        v1 = max(0, int(y1 * scale_v))
        u2 = min(256, int(x2 * scale_u))
        v2 = min(256, int(y2 * scale_v))

        if u2 <= u1 or v2 <= v1:
            return 25.0

        # Query center region of the vehicle box
        cu_margin = int((u2 - u1) * 0.20)
        cv_margin = int((v2 - v1) * 0.20)

        crop = inv_depth_map[v1 + cv_margin:v2 - cv_margin, u1 + cu_margin:u2 - cu_margin]
        if crop.size == 0:
            crop = inv_depth_map[v1:v2, u1:u2]

        median_inv_depth = float(np.median(crop))

        # Calibrated scale: 4300.0 / median_inv_depth
        if median_inv_depth <= 10.0:
            metric_depth_m = 65.0
        else:
            metric_depth_m = float(4300.0 / (median_inv_depth + 1e-3))

        return max(5.0, min(80.0, metric_depth_m))
