#!/usr/bin/env python3
"""
Memory-safe OpenLane cache generator for Kaggle.

Official tools/convert_datasets/openlane.py --generate keeps ALL frames in a
giant dict, then writes pickles → OOM around ~120k frames on ~30GB RAM.

This script:
  - processes JSONL in CHUNK_SIZE batches
  - writes .pkl immediately per batch (releases RAM)
  - resumes by skipping existing .pkl files
  - rebuilds data_lists/*.txt at the end

Run inside Kaggle with the lane3d env, e.g.:
  /kaggle/working/miniconda3/envs/lane3d/bin/python generate_cache_chunked.py
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import mmcv
import numpy as np
import tqdm

# Anchor3DLane must be on PYTHONPATH / cwd
sys.path.insert(0, os.environ.get("ANCHOR3DLANE_ROOT", "/kaggle/working/Anchor3DLane"))

from tools.convert_datasets.openlane import (  # noqa: E402
    transform_annotation,
    lane_smoothing,
    make_lane_y_mono_inc,
)
from mmseg.datasets.tools.utils import (  # noqa: E402
    prune_3d_lane_by_visibility,
    prune_3d_lane_by_range,
    resample_laneline_in_y,
)


def process_one_line(info_dict, data_root, max_lanes=20, sample_step=1, prune_vis=True, smooth=True, test_mode=False):
    """Return annotation dict or None if skipped (same filters as official extract)."""
    image_path = os.path.join("images", info_dict["file_path"])
    full_img = os.path.join(data_root, image_path)
    if not os.path.exists(full_img):
        return None

    cam_extrinsics = np.array(info_dict["extrinsic"])
    cam_intrinsics = np.array(info_dict["intrinsic"])

    R_vg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    R_gc = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    cam_extrinsics = cam_extrinsics.copy()
    cam_extrinsics[:3, :3] = np.matmul(
        np.matmul(np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]), R_vg), R_gc
    )
    cam_extrinsics[0:2, 3] = 0.0

    gt_lanes_packed = info_dict.get("lane_lines", [])
    if len(gt_lanes_packed) < 1:
        if test_mode:
            return {
                "path": image_path,
                "gt_3dlanes": [],
                "categories": [],
                "aug": False,
                "relative_path": info_dict["file_path"],
                "gt_camera_extrinsic": cam_extrinsics,
                "gt_camera_intrinsic": cam_intrinsics,
                "json_line": info_dict,
            }
        return None

    anchor_y_steps = np.linspace(1, 200, 200 // sample_step)
    all_lanes, lane_cates = [], []

    for gt_lane_packed in gt_lanes_packed:
        lane = np.array(gt_lane_packed["xyz"])
        lane_visibility = np.array(gt_lane_packed["visibility"])
        lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
        cam_representation = np.linalg.inv(
            np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
        )
        lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
        lane = lane[0:3, :].T

        if prune_vis:
            lane = prune_3d_lane_by_visibility(lane, lane_visibility)
        pruned_lane = prune_3d_lane_by_range(lane, -30, 30)
        if pruned_lane.shape[0] < 2:
            continue
        pruned_lane = make_lane_y_mono_inc(pruned_lane)
        if pruned_lane.shape[0] < 2:
            continue
        if smooth:
            pruned_lane = lane_smoothing(pruned_lane)
            if pruned_lane.shape[0] < 2:
                continue
        if (pruned_lane[-1, 1] - pruned_lane[0, 1]) < 5:
            continue

        x_values, z_values, visibility_vec = resample_laneline_in_y(
            pruned_lane, anchor_y_steps, out_vis=True
        )
        if sum(visibility_vec) <= 1:
            continue
        resample_lane = np.stack([x_values, z_values, visibility_vec], axis=-1)
        cate = gt_lane_packed["category"]
        if cate == 21:
            cate = 20
        all_lanes.append({"gt_lane": resample_lane, "category": cate})
        lane_cates.append(cate)

    if len(all_lanes) == 0 or len(all_lanes) > max_lanes or max(lane_cates) > 20:
        if test_mode:
            return {
                "path": image_path,
                "gt_3dlanes": [],
                "categories": [],
                "aug": False,
                "relative_path": info_dict["file_path"],
                "gt_camera_extrinsic": cam_extrinsics,
                "gt_camera_intrinsic": cam_intrinsics,
                "json_line": info_dict,
            }
        return None

    return {
        "path": image_path,
        "gt_3dlanes": [p["gt_lane"] for p in all_lanes],
        "categories": [p["category"] for p in all_lanes],
        "aug": False,
        "relative_path": info_dict["file_path"],
        "gt_camera_extrinsic": cam_extrinsics,
        "gt_camera_intrinsic": cam_intrinsics,
        "json_line": info_dict,
    }


def write_pickle(old_anno, tar_path, max_lanes=20):
    new_anno = transform_annotation(old_anno, max_lanes=max_lanes, anchor_len=200)
    anno = {
        "filename": new_anno["path"],
        "gt_3dlanes": new_anno["gt_3dlanes"],
        "gt_camera_extrinsic": new_anno["gt_camera_extrinsic"],
        "gt_camera_intrinsic": new_anno["gt_camera_intrinsic"],
        "old_anno": new_anno["old_anno"],
    }
    rel = "/".join(anno["filename"].split("/")[-3:]).replace(".jpg", ".pkl")
    pickle_file = os.path.join(tar_path, rel)
    mmcv.mkdir_or_exist(os.path.dirname(pickle_file))
    if os.path.exists(pickle_file):
        return pickle_file, True  # already exists
    with open(pickle_file, "wb") as w:
        pickle.dump(
            {
                "image_id": anno["filename"],
                "gt_3dlanes": anno["gt_3dlanes"],
                "gt_camera_extrinsic": anno["gt_camera_extrinsic"],
                "gt_camera_intrinsic": anno["gt_camera_intrinsic"],
            },
            w,
        )
    return pickle_file, False


def generate_split(
    data_root: str,
    split_json: str,
    tar_path: str,
    chunk_size: int = 5000,
    max_lanes: int = 20,
    test_mode: bool = False,
    max_frames: int | None = None,
):
    import json

    written, skipped, missing_img, existed = 0, 0, 0, 0
    batch = []

    def flush(batch_annos):
        nonlocal written, existed
        for anno in batch_annos:
            _, already = write_pickle(anno, tar_path, max_lanes=max_lanes)
            if already:
                existed += 1
            else:
                written += 1
        batch_annos.clear()

    with open(split_json, "r") as f:
        for i, line in enumerate(tqdm.tqdm(f, desc=os.path.basename(split_json))):
            if max_frames is not None and i >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            info = json.loads(line)
            # resume: skip if pkl exists
            image_path = os.path.join("images", info["file_path"])
            rel_pkl = "/".join(image_path.split("/")[-3:]).replace(".jpg", ".pkl")
            if os.path.exists(os.path.join(tar_path, rel_pkl)):
                existed += 1
                continue

            anno = process_one_line(
                info, data_root, max_lanes=max_lanes, test_mode=test_mode
            )
            if anno is None:
                # distinguish missing image vs filtered
                if not os.path.exists(os.path.join(data_root, image_path)):
                    missing_img += 1
                else:
                    skipped += 1
                continue
            batch.append(anno)
            if len(batch) >= chunk_size:
                flush(batch)

        if batch:
            flush(batch)

    print(
        f"done {split_json}: newly_written={written} already_existed={existed} "
        f"filtered={skipped} missing_img={missing_img}"
    )


def rebuild_lists(tar_path: str, data_list_path: str):
    os.makedirs(data_list_path, exist_ok=True)
    for split in ("training", "validation"):
        cache = os.path.join(tar_path, split)
        out = os.path.join(data_list_path, f"{split}.txt")
        files = sorted(glob.glob(os.path.join(cache, "seg*", "*.pkl")))
        with open(out, "w") as w:
            for item in files:
                name = "/".join(item[:-4].split("/")[-3:])
                w.write(name + "\n")
        print(f"{split}: {len(files)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/kaggle/working/data/OpenLane")
    ap.add_argument("--cache-dir", default="/tmp/cache_dense")
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--max-frames", type=int, default=None, help="optional cap for testing")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()

    data_root = args.data_root
    tar_path = args.cache_dir
    os.makedirs(tar_path, exist_ok=True)

    # ensure symlink for training code
    link = os.path.join(data_root, "cache_dense")
    if os.path.lexists(link):
        if os.path.islink(link):
            os.remove(link)
        elif os.path.isdir(link) and os.path.abspath(link) != os.path.abspath(tar_path):
            # do not delete real dirs accidentally; just warn
            print("WARNING: cache_dense exists and is not a symlink:", link)
    if not os.path.lexists(link):
        os.symlink(tar_path, link)
        print("linked", link, "->", tar_path)

    train_json = os.path.join(data_root, "data_splits", "training.json")
    val_json = os.path.join(data_root, "data_splits", "validation.json")
    assert os.path.exists(train_json), train_json

    generate_split(
        data_root,
        train_json,
        tar_path,
        chunk_size=args.chunk_size,
        test_mode=False,
        max_frames=args.max_frames,
    )
    if not args.train_only and os.path.exists(val_json) and os.path.getsize(val_json) > 0:
        generate_split(
            data_root,
            val_json,
            tar_path,
            chunk_size=args.chunk_size,
            test_mode=True,
            max_frames=args.max_frames,
        )

    rebuild_lists(tar_path, os.path.join(data_root, "data_lists"))
    n = len(glob.glob(os.path.join(tar_path, "**", "*.pkl"), recursive=True))
    print("TOTAL pkls:", n)


if __name__ == "__main__":
    main()
