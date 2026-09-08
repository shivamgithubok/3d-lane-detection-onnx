"""
Lane-only live TensorRT viewer with Garmin HUD crop + letterbox to 480x360.
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.inference.postprocess import (
    ANCHOR_Y_STEPS,
    ANCHOR_LEN,
    postprocess_onnx_output,
    decode_lane_pixels,
)
from src.utils.visualization import draw_bev

ENGINE_PATH = "models/anchor3dlane_raw.engine"
IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480

P_MATRIX = np.array(
    [
        [503.75, 239.67108834, 12.5606295, 0.0],
        [0.0, 181.326628, -557.993558, 850.078125],
        [0.0, 0.998629535, 0.0523359562, 0.0],
    ]
)

LANE_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 128, 0),
    (255, 0, 255),
    (255, 255, 0),
    (100, 100, 255),
]


def detect_hud_top(frame, min_dark_frac=0.55, min_hud_h=20):
    """Find y where Garmin black HUD bar starts (scan from bottom)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    hud_rows = []
    for y in range(h - 1, -1, -1):
        if (gray[y] < 40).mean() > min_dark_frac:
            hud_rows.append(y)
        elif hud_rows:
            break
    if len(hud_rows) < min_hud_h:
        # fallback: fixed ~7% for Garmin 540p
        return int(h * 0.93)
    return min(hud_rows)


def crop_garmin(frame, hud_top=None, hood_frac=0.08):
    """Remove bottom HUD (+ optional hood fraction of remaining content)."""
    h = frame.shape[0]
    if hud_top is None:
        hud_top = detect_hud_top(frame)
    content = frame[:hud_top]
    if hood_frac > 0:
        cut = int(content.shape[0] * (1.0 - hood_frac))
        content = content[: max(cut, 1)]
    return content, hud_top


def letterbox(frame, out_w=INPUT_W, out_h=INPUT_H, pad_value=0):
    """Resize keeping aspect ratio, pad to out_w x out_h (no stretch)."""
    h, w = frame.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), pad_value, dtype=frame.dtype)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def preprocess(frame, crop=True, hood_frac=0.08, use_letterbox=True, hud_top=None):
    meta = {"hud_top": None, "crop_shape": None, "letterbox": use_letterbox}
    work = frame
    if crop:
        work, hud_top = crop_garmin(frame, hud_top=hud_top, hood_frac=hood_frac)
        meta["hud_top"] = hud_top
        meta["crop_shape"] = work.shape[:2]
    if use_letterbox:
        resized, scale, x0, y0 = letterbox(work)
        meta.update({"scale": scale, "pad_x": x0, "pad_y": y0})
    else:
        resized = cv2.resize(work, (INPUT_W, INPUT_H))
        meta.update({"scale": None, "pad_x": 0, "pad_y": 0})
    img = resized[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...].astype(np.float32))
    mask = np.ascontiguousarray(np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32))
    return img, mask, resized, work, meta


def lane_stats(proposals, scores):
    rows = []
    if proposals is None or len(proposals) == 0:
        return rows
    for i, p in enumerate(proposals):
        vis = p[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
        xs = p[5 : 5 + ANCHOR_LEN][vis]
        mean_x = float(np.mean(xs)) if vis.sum() else float("nan")
        sc = float(scores[i]) if scores is not None else float(p[1])
        rows.append((i, sc, mean_x, int(vis.sum())))
    return rows


def draw_lanes_labeled(frame, proposals, scores):
    out = frame.copy()
    if proposals is None:
        return out
    for i, lane in enumerate(proposals):
        color = LANE_COLORS[i % len(LANE_COLORS)]
        pts = decode_lane_pixels(lane, P_MATRIX)
        draw_pts = [
            (int(u), int(v))
            for u, v in pts
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]
        ]
        for j in range(1, len(draw_pts)):
            cv2.line(out, draw_pts[j - 1], draw_pts[j], color, 2, cv2.LINE_AA)
        if draw_pts:
            sc = float(scores[i]) if scores is not None else float(lane[1])
            vis = lane[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
            xs = lane[5 : 5 + ANCHOR_LEN][vis]
            mean_x = float(np.mean(xs)) if vis.sum() else 0.0
            label = f"L{i} s={sc:.2f} x={mean_x:+.1f}m"
            cv2.putText(
                out, label, draw_pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )
    return out


def export_cropped_video(src, dst, hud_top, hood_frac=0.08):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read {src}")
    crop0, _ = crop_garmin(frame0, hud_top=hud_top, hood_frac=hood_frac)
    ch, cw = crop0.shape[:2]
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    writer.write(crop0)
    n = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop, _ = crop_garmin(frame, hud_top=hud_top, hood_frac=hood_frac)
        if crop.shape[0] != ch or crop.shape[1] != cw:
            crop = cv2.resize(crop, (cw, ch))
        writer.write(crop)
        n += 1
    cap.release()
    writer.release()
    return n, (cw, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="testing_new_videos/GRMN6694_540.mp4")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--crop-only", action="store_true", help="write *_nohud.mp4 and exit")
    ap.add_argument("--no-letterbox", action="store_true")
    ap.add_argument("--hood-frac", type=float, default=0.08)
    ap.add_argument("--hud-top", type=int, default=-1, help="fixed y cut; -1 = auto")
    args = ap.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Missing video: {video_path}")
        sys.exit(1)

    # Probe HUD once
    cap0 = cv2.VideoCapture(video_path)
    ok, frame0 = cap0.read()
    cap0.release()
    hud_top = args.hud_top if args.hud_top >= 0 else detect_hud_top(frame0)
    print(f"HUD crop y<{hud_top} (remove bottom {frame0.shape[0]-hud_top}px), hood_frac={args.hood_frac}")

    os.makedirs("output", exist_ok=True)
    stem, ext = os.path.splitext(video_path)
    cropped_path = f"{stem}_nohud{ext or '.mp4'}"
    if not args.no_crop:
        print(f"Writing cropped video -> {cropped_path}")
        n, wh = export_cropped_video(video_path, cropped_path, hud_top, args.hood_frac)
        print(f"Cropped {n} frames to {wh[0]}x{wh[1]}")
        if args.crop_only:
            return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"Loading {ENGINE_PATH} ...")
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    d_img = cuda.mem_alloc(1 * 3 * INPUT_H * INPUT_W * 4)
    d_mask = cuda.mem_alloc(1 * 1 * INPUT_H * INPUT_W * 4)
    h_reg = np.empty((1, 4431, 86), dtype=np.float32)
    h_anc = np.empty((1, 4431, 65), dtype=np.float32)
    d_reg = cuda.mem_alloc(h_reg.nbytes)
    d_anc = cuda.mem_alloc(h_anc.nbytes)
    stream = cuda.Stream()
    context.set_tensor_address("img", int(d_img))
    context.set_tensor_address("mask", int(d_mask))
    context.set_tensor_address("reg_proposals", int(d_reg))
    context.set_tensor_address("anchors", int(d_anc))

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {video_path} {w}x{h} @ {fps_v:.1f} fps, {nframes} frames")
    print("Press q to quit.")

    out_path = os.path.join("output", f"{os.path.basename(stem)}_lanes_nohud.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), min(30.0, fps_v), (960, 720)
    )

    cv2.namedWindow("Lane-only Front (HUD cropped)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Lane-only BEV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane-only Front (HUD cropped)", 960, 720)
    cv2.resizeWindow("Lane-only BEV", 480, 720)

    fps_hist = []
    lane_counts = []
    score_means = []
    empty = 0
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        img, mask, model_img, crop_bgr, meta = preprocess(
            frame,
            crop=not args.no_crop,
            hood_frac=args.hood_frac,
            use_letterbox=not args.no_letterbox,
            hud_top=hud_top,
        )
        cuda.memcpy_htod_async(d_img, img, stream)
        cuda.memcpy_htod_async(d_mask, mask, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_reg, d_reg, stream)
        cuda.memcpy_dtoh_async(h_anc, d_anc, stream)
        stream.synchronize()
        proposals, scores = postprocess_onnx_output(h_reg)
        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_hist.append(fps)

        n = 0 if proposals is None else len(proposals)
        lane_counts.append(n)
        if n == 0:
            empty += 1
        else:
            score_means.append(float(np.mean(scores)) if scores is not None else 0.0)

        front = draw_lanes_labeled(model_img, proposals, scores)
        # side-by-side: cropped source (scaled) | model input with lanes
        crop_show = cv2.resize(crop_bgr, (480, 360))
        panel = np.hstack([crop_show, front])
        bev = draw_bev(proposals, ANCHOR_Y_STEPS)
        display = cv2.resize(panel, (960, 720))
        avg = float(np.mean(fps_hist[-30:]))
        hud1 = f"HUD-CROP+LETTERBOX  {avg:.1f} FPS  {dt*1000:.0f}ms  lanes={n}"
        hud2 = f"src {w}x{h} crop={meta.get('crop_shape')} -> 480x360 letterbox={not args.no_letterbox}"
        cv2.putText(display, hud1, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, hud2, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y = 98
        for i, sc, mean_x, nvis in lane_stats(proposals, scores):
            color = LANE_COLORS[i % len(LANE_COLORS)]
            cv2.putText(
                display,
                f"L{i}: score={sc:.2f} mean_x={mean_x:+.1f}m vis={nvis}",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 22

        writer.write(display)
        cv2.imshow("Lane-only Front (HUD cropped)", display)
        cv2.imshow("Lane-only BEV", bev)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit by user")
            break

        frame_i += 1
        if frame_i % 30 == 0:
            print(
                f"f{frame_i} lanes={n} fps={avg:.1f} scores="
                f"{[round(float(s), 2) for s in (scores if scores is not None else [])]}"
            )

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n===== LANE-ONLY AFTER HUD CROP =====")
    print(f"frames={frame_i} empty={empty} ({100*empty/max(frame_i,1):.1f}%)")
    if lane_counts:
        print(
            f"lanes/frame mean={np.mean(lane_counts):.2f} "
            f"median={np.median(lane_counts):.1f} max={np.max(lane_counts)}"
        )
    if score_means:
        print(f"mean score when lanes present={np.mean(score_means):.3f}")
    print(f"avg GPU FPS={np.mean(fps_hist):.2f}")
    print(f"saved {out_path}")
    if not args.no_crop:
        print(f"cropped source saved {cropped_path}")


if __name__ == "__main__":
    main()
