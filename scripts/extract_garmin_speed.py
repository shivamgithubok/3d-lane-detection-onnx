#!/usr/bin/env python3
"""Extract Garmin HUD speed (NN MPH) to a per-frame JSON log.

Example:
  python scripts/extract_garmin_speed.py testing_new_videos/GRMN6694_540.mp4
  # writes testing_new_videos/GRMN6694_540_speed.json

The nohud clip uses the same frame index:
  python scripts/infer_video.py testing_new_videos/GRMN6694_540_nohud.mp4
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.ego_speed import default_speed_json_path, extract_speed_log, save_speed_log


def main():
    parser = argparse.ArgumentParser(description="OCR Garmin HUD speed → JSON")
    parser.add_argument("video", help="With-HUD Garmin mp4 (not *_nohud)")
    parser.add_argument("--output", default="", help="JSON path (default: <stem>_speed.json)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}")
        return 1
    out = args.output or default_speed_json_path(args.video)
    print(f"Reading HUD speed from {args.video} ...")
    log = extract_speed_log(args.video, max_frames=args.max_frames)
    save_speed_log(log, out)
    vals = [v for v in log["mph"] if v is not None]
    print(f"Wrote {out}")
    print(f"  frames={log['frame_count']} valid={log['valid_frames']} fps={log['fps']:.2f}")
    if vals:
        print(f"  mph min/max/last = {min(vals)} / {max(vals)} / {vals[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
