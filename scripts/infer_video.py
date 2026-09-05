import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.postprocess import postprocess_onnx_output, decode_lane_pixels, ANCHOR_Y_STEPS
from src.inference import lane_filter_config as lane_cfg
from src.inference.lane_preprocess import prepare_lane_input
from src.utils.visualization import draw_bev
from src.utils.drivable_area import get_ego_corridor_2d_pixels
from src.tracking.lane_association import LaneTrackerManager
from src.tracking.road_state import RoadStateEstimator
from src.utils.ego_speed import EgoSpeedLog

MODEL_PATH = "models/anchor3dlane_raw.onnx"
ENGINE_PATH = "models/anchor3dlane_raw.engine"
DEFAULT_VIDEO_PATH = "testing_new_videos/GRMN6694_540_nohud.mp4"
OUTPUT_VIDEO_PATH = "output/infer_video_ekf_annotated.mp4"

INPUT_H, INPUT_W = 360, 480

P_MATRIX = np.array([
    [503.75, 239.67108834, 12.5606295, 0.0],
    [0.0, 181.326628, -557.993558, 850.078125],
    [0.0, 0.998629535, 0.0523359562, 0.0]
])

def load_lane_backend(force_onnx=False):
    """Prefer TensorRT engine; fall back to ONNX Runtime."""
    use_engine = (not force_onnx) and os.path.exists(ENGINE_PATH)
    if use_engine:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        logger = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        dummy_img = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        dummy_mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)
        h_reg = np.empty((1, 4431, 86), dtype=np.float32)
        h_anc = np.empty((1, 4431, 65), dtype=np.float32)
        d_img = cuda.mem_alloc(dummy_img.nbytes)
        d_mask = cuda.mem_alloc(dummy_mask.nbytes)
        d_reg = cuda.mem_alloc(h_reg.nbytes)
        d_anc = cuda.mem_alloc(h_anc.nbytes)
        stream = cuda.Stream()
        context.set_tensor_address("img", int(d_img))
        context.set_tensor_address("mask", int(d_mask))
        context.set_tensor_address("reg_proposals", int(d_reg))
        context.set_tensor_address("anchors", int(d_anc))

        def infer(img, mask):
            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)
            cuda.memcpy_htod_async(d_img, img, stream)
            cuda.memcpy_htod_async(d_mask, mask, stream)
            context.execute_async_v3(stream.handle)
            cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
            cuda.memcpy_dtoh_async(h_anc, d_anc, stream)
            stream.synchronize()
            return h_reg, h_anc

        print(f"Lane backend: TensorRT ({ENGINE_PATH})")
        return infer

    providers = ort.get_available_providers()
    print("Available execution providers:", providers)
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)

    def infer(img, mask):
        return sess.run(None, {"img": img, "mask": mask})

    print(f"Lane backend: ONNX Runtime ({MODEL_PATH})")
    return infer


def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img, mask, meta = prepare_lane_input(resized)
    return img, mask, resized, meta

def project_3d_points(pts_3d, P_matrix):
    xs = pts_3d[:, 0]
    ys = pts_3d[:, 1]
    zs = pts_3d[:, 2]
    ones = np.ones((1, len(zs)))
    coords = np.vstack((xs, ys, zs, ones))
    trans = P_matrix @ coords
    u = trans[0, :] / (trans[2, :] + 1e-8)
    v = trans[1, :] / (trans[2, :] + 1e-8)
    return np.column_stack((u, v))

def draw_lanes(frame, proposals, color=(0, 255, 255), thickness=2):
    if proposals is None or len(proposals) == 0:
        return frame
    for lane in proposals:
        if isinstance(lane, np.ndarray) and lane.ndim == 2 and lane.shape[1] == 3:
            pts = project_3d_points(lane, P_MATRIX)
        else:
            pts = decode_lane_pixels(lane, P_MATRIX)
        draw_pts = [(int(u), int(v)) for u, v in pts if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]]
        for i in range(1, len(draw_pts)):
            cv2.line(frame, draw_pts[i-1], draw_pts[i], color, thickness)
    return frame

def main():
    parser = argparse.ArgumentParser(description="3D Lane Detection with EKF Temporal Tracking")
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO_PATH, help="Path to input video file")
    parser.add_argument("--no-ekf", action="store_true", help="Disable EKF temporal tracking (run raw per-frame)")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated video to output directory")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save output video")
    parser.add_argument("--output", default=OUTPUT_VIDEO_PATH, help="Path for saved video")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without GUI windows")
    parser.add_argument("--no-onesided", action="store_true", help="Disable one-sided ego reconstruct (P1)")
    parser.add_argument("--compare", action="store_true", help="A/B onesided on vs off on the same detections")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    parser.add_argument("--onnx", action="store_true", help="Force ONNX Runtime instead of TensorRT engine")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return

    infer_lanes = load_lane_backend(force_onnx=args.onnx)
    cap = cv2.VideoCapture(args.video)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / max(1e-3, float(source_fps))
    speed_log = EgoSpeedLog.auto_load(args.video)
    if speed_log is not None:
        print(f"Ego speed JSON: {len(speed_log.mps)} frames (HUD MPH log)")
    else:
        print("Ego speed JSON: not found (EKF will coast without v)")

    use_ekf = not args.no_ekf
    if args.no_onesided:
        lane_cfg.ENABLE_ONESIDED_RECONSTRUCT = False

    tracker_manager = None
    road_state_estimator = None
    compare_estimator = None
    if use_ekf:
        print("Initializing EKF + ego road-state tracker...")
        road_state_estimator = RoadStateEstimator()
        if args.compare:
            compare_estimator = RoadStateEstimator()
    else:
        tracker_manager = LaneTrackerManager(
            max_missed_frames=lane_cfg.EKF_MAX_MISSED_FRAMES,
            dist_threshold=lane_cfg.EKF_DIST_THRESHOLD_M,
            confirm_hits=lane_cfg.EKF_CONFIRM_HITS,
            require_confirmed=lane_cfg.EKF_REQUIRE_CONFIRMED,
        )

    show_gui = not args.no_gui
    if show_gui:
        try:
            cv2.namedWindow("Front View", cv2.WINDOW_NORMAL)
            cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
        except Exception:
            show_gui = False

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, source_fps if source_fps > 1 else 30.0, (960, 720))

    frame_count = 0
    fps_history = []
    ekf_time_history = []
    stats_on = {"CONFIRMED": 0, "PREDICTED": 0, "UNKNOWN": 0, "onesided": 0, "corridor": 0}
    stats_off = {"CONFIRMED": 0, "PREDICTED": 0, "UNKNOWN": 0, "onesided": 0, "corridor": 0}

    onesided_on = bool(lane_cfg.ENABLE_ONESIDED_RECONSTRUCT)
    print(
        f"Starting inference on '{args.video}' | EKF: {'ON' if use_ekf else 'OFF'} | "
        f"onesided reconstruct: {'ON' if onesided_on else 'OFF'} | "
        f"P3 CLAHE: {'ALWAYS' if lane_cfg.ENABLE_CLAHE_ALWAYS else ('gated' if lane_cfg.ENABLE_DARK_CLAHE else 'off')} "
        f"adapt_conf={'ON' if lane_cfg.ENABLE_ADAPTIVE_CONF else 'OFF'}"
    )

    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        img, mask, resized, prep_meta = preprocess(frame)

        # 1. Neural Network Inference
        reg_proposals, anchors = infer_lanes(img, mask)
        raw_proposals, scores = postprocess_onnx_output(
            reg_proposals, conf_threshold=prep_meta["conf"]
        )

        # 2. EKF + road-state (onesided reconstruct lives here)
        t_ekf_start = time.perf_counter()
        road_state = None
        compare_state = None
        speed_mps = speed_log.get_mps(frame_count) if speed_log is not None else None
        if use_ekf and road_state_estimator is not None:
            if args.compare and compare_estimator is not None:
                prev_flag = bool(lane_cfg.ENABLE_ONESIDED_RECONSTRUCT)
                lane_cfg.ENABLE_ONESIDED_RECONSTRUCT = False
                compare_state = compare_estimator.update(
                    raw_proposals, dt=dt, speed_mps=speed_mps
                )
                lane_cfg.ENABLE_ONESIDED_RECONSTRUCT = prev_flag
            road_state = road_state_estimator.update(
                raw_proposals, dt=dt, speed_mps=speed_mps
            )
            active_lanes = road_state.visual_lanes
        elif use_ekf and tracker_manager is not None:
            active_lanes = tracker_manager.update(
                raw_proposals, dt=dt, speed_mps=speed_mps
            )
        else:
            active_lanes = raw_proposals
        t_ekf_end = time.perf_counter()
        ekf_time_history.append((t_ekf_end - t_ekf_start) * 1000.0)

        def _tally(bucket, state):
            if state is None:
                return
            bucket[state.status] = bucket.get(state.status, 0) + 1
            if state.has_valid_corridor:
                bucket["corridor"] += 1
            if state.source.startswith("onesided"):
                bucket["onesided"] += 1

        _tally(stats_on, road_state)
        _tally(stats_off, compare_state)

        t1 = time.perf_counter()
        dt_total = t1 - t0
        fps = 1.0 / dt_total if dt_total > 0 else 0.0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history[-30:])
        avg_ekf_ms = np.mean(ekf_time_history[-30:])

        # 3. Visualization
        lane_color = (0, 255, 255) if use_ekf else (0, 0, 255)
        front_annotated = draw_lanes(resized.copy(), active_lanes, color=lane_color, thickness=2)
        if road_state is not None and road_state.reconstructed_side is not None:
            recon_lane = road_state.ego_right if road_state.reconstructed_side == "right" else road_state.ego_left
            if recon_lane is not None:
                front_annotated = draw_lanes(
                    front_annotated, [recon_lane], color=(0, 165, 255), thickness=2
                )
        if road_state is not None and road_state.has_valid_corridor:
            poly = get_ego_corridor_2d_pixels(
                active_lanes,
                P_MATRIX,
                img_size=(INPUT_W, INPUT_H),
                target_size=(INPUT_W, INPUT_H),
                ego_left=road_state.ego_left,
                ego_right=road_state.ego_right,
                left_corridor_3d=road_state.left_corridor_3d,
                right_corridor_3d=road_state.right_corridor_3d,
            )
            if poly is not None:
                overlay = front_annotated.copy()
                fill = (80, 220, 80) if road_state.is_confirmed else (40, 180, 255)
                cv2.fillPoly(overlay, [poly], fill)
                cv2.addWeighted(overlay, 0.28, front_annotated, 0.72, 0, front_annotated)
            bev = draw_bev(
                active_lanes,
                ANCHOR_Y_STEPS,
                left_corridor_3d=road_state.left_corridor_3d,
                right_corridor_3d=road_state.right_corridor_3d,
                allow_auto_corridor=False,
            )
        else:
            bev = draw_bev(active_lanes, ANCHOR_Y_STEPS, allow_auto_corridor=False)

        display_frame = cv2.resize(front_annotated, (960, 720))
        if road_state is not None:
            status = road_state.status
            src = road_state.source
            wlock = road_state.locked_width_m
            wtxt = f"{wlock:.2f}m" if wlock is not None else "--"
            color = (80, 220, 80) if status == "CONFIRMED" else (
                (40, 180, 255) if status == "PREDICTED" else (60, 60, 220)
            )
            mode_text = f"{status} | {src} | W={wtxt}"
        else:
            color = (255, 255, 0)
            mode_text = f"EKF Mode: {'ON (Smoothed Memory)' if use_ekf else 'OFF (Raw Detection)'}"
        cv2.putText(display_frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f} ({dt_total*1000:.1f}ms) | EKF Math: {avg_ekf_ms:.3f}ms",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        n_lanes = len(active_lanes) if active_lanes is not None else 0
        cv2.putText(display_frame, f"Active Lanes: {n_lanes}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(display_frame)

        if show_gui:
            try:
                cv2.imshow("Front View", display_frame)
                cv2.imshow("BEV", bev)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass

        frame_count += 1
        if frame_count % 30 == 0:
            st = road_state.status if road_state is not None else "-"
            src = road_state.source if road_state is not None else "-"
            print(
                f"Frame {frame_count:4d} | FPS: {avg_fps:.1f} | {st}/{src} | "
                f"Active Lanes: {len(active_lanes) if active_lanes is not None else 0}"
            )
        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if show_gui:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"\nFinished processing {frame_count} frames.")
    if fps_history:
        print(f"Overall Average Pipeline FPS: {np.mean(fps_history):.2f}")
    if ekf_time_history:
        print(f"Average EKF Tracking Math Overhead: {np.mean(ekf_time_history):.4f} ms per frame")

    def _pct(n):
        return 100.0 * n / max(1, frame_count)

    print("\nRoad-state coverage (onesided ON):")
    print(
        f"  CONFIRMED {stats_on['CONFIRMED']} ({_pct(stats_on['CONFIRMED']):.1f}%)  "
        f"PREDICTED {stats_on['PREDICTED']} ({_pct(stats_on['PREDICTED']):.1f}%)  "
        f"UNKNOWN {stats_on['UNKNOWN']} ({_pct(stats_on['UNKNOWN']):.1f}%)"
    )
    print(
        f"  corridor visible {stats_on['corridor']} ({_pct(stats_on['corridor']):.1f}%)  "
        f"onesided reconstruct {stats_on['onesided']} ({_pct(stats_on['onesided']):.1f}%)"
    )
    if args.compare:
        print("Road-state coverage (onesided OFF, same detections):")
        print(
            f"  CONFIRMED {stats_off['CONFIRMED']} ({_pct(stats_off['CONFIRMED']):.1f}%)  "
            f"PREDICTED {stats_off['PREDICTED']} ({_pct(stats_off['PREDICTED']):.1f}%)  "
            f"UNKNOWN {stats_off['UNKNOWN']} ({_pct(stats_off['UNKNOWN']):.1f}%)"
        )
        print(
            f"  corridor visible {stats_off['corridor']} ({_pct(stats_off['corridor']):.1f}%)  "
            f"onesided reconstruct {stats_off['onesided']} ({_pct(stats_off['onesided']):.1f}%)"
        )
        delta = stats_on["corridor"] - stats_off["corridor"]
        print(f"  corridor frames gained by P1: {delta} ({_pct(delta):+.1f} pp)")
    if args.save:
        print(f"Saved annotated video to: {args.output}")

if __name__ == "__main__":
    main()
    