"""
Generates an image classification dataset for YOLOv8/YOLO26.
Extracts crops of the sink zones from a full video.
Labels them 'washing' (if inside GT AND that specific station has motion)
or 'not_washing'.

Usage:
    python src/generate_yolo_dataset.py <video> <gt_json> [--roi roi.json] [--out datasets/yolo_cls] [--val-ratio 0.2]
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path

from config import DEFAULT_ROI_PATH, PROJECT_ROOT

# Parameters
FPS_EXTRACT_WASHING = 2.0      # frames per second to extract during washing
FPS_EXTRACT_NOT_WASHING = 0.2  # less frames for not washing to balance dataset
PAD_X, PAD_Y = 150, 100        # padding around sink zones to capture arms/water
MOTION_THRESH = 500            # minimum moving pixels to consider crop active


def get_padded_crop(frame, zone):
    h, w = frame.shape[:2]
    zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
    x1 = max(0, zx - PAD_X)
    y1 = max(0, zy - PAD_Y)
    x2 = min(w, zx + zw + PAD_X)
    y2 = min(h, zy + zh + PAD_Y)
    return frame[y1:y2, x1:x2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate YOLO classification dataset from annotated video."
    )
    parser.add_argument("video", help="Path to input video.")
    parser.add_argument("gt_json", help="Path to ground-truth JSON (from annotate_full.py).")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH),
                        help="Path to ROI JSON (default: outputs/roi.json).")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "datasets" / "yolo_cls"),
                        help="Output dataset directory (default: datasets/yolo_cls).")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of video used for validation (from the end). Default 0.2.")
    parser.add_argument("--prefix", default=None,
                        help="Filename prefix for saved images (default: auto from video name).")
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    gt_path = Path(args.gt_json)
    roi_path = Path(args.roi)
    out_dir = Path(args.out)

    # Create dirs (append-safe, no rmtree)
    for split in ("train", "val"):
        for label in ("washing", "not_washing"):
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Load data
    gt_data = json.load(open(gt_path, encoding="utf-8"))
    events = gt_data["events"]
    print(f"Loaded {len(events)} GT events from {gt_path.name}")

    roi_data = json.load(open(roi_path, encoding="utf-8"))
    sink_zones = roi_data["sink_zones"]
    print(f"Loaded {len(sink_zones)} sink zones from {roi_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps
    val_start_sec = total_sec * (1.0 - args.val_ratio)

    frame_interval_wash = max(1, int(fps / FPS_EXTRACT_WASHING))
    frame_interval_not_wash = max(1, int(fps / FPS_EXTRACT_NOT_WASHING))

    # Per-station background subtractors for motion detection
    bg_subs = [
        cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
        for _ in sink_zones
    ]

    # Filename prefix
    prefix = args.prefix or video_path.stem

    frame_idx = 0
    saved_wash = 0
    saved_not_wash = 0

    print(f"Processing {video_path.name} ({total_sec/60:.1f} min, {fps:.0f} fps)...")
    print(f"Train/val split at {val_start_sec/60:.1f} min (val ratio {args.val_ratio})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_sec = frame_idx / fps
        is_in_gt = any(ev["start_sec"] <= current_sec <= ev["end_sec"] for ev in events)

        interval = frame_interval_wash if is_in_gt else frame_interval_not_wash

        if frame_idx % interval == 0:
            split = "val" if current_sec >= val_start_sec else "train"

            for i, (zone, bg_sub) in enumerate(zip(sink_zones, bg_subs)):
                crop = get_padded_crop(frame, zone)
                mask = bg_sub.apply(crop)
                motion_pixels = cv2.countNonZero(mask)

                # Washing ONLY if GT says so AND this specific station has motion
                if is_in_gt and motion_pixels > MOTION_THRESH:
                    label = "washing"
                else:
                    label = "not_washing"

                out_file = out_dir / split / label / f"{prefix}_st{i}_{frame_idx:07d}.jpg"
                cv2.imwrite(str(out_file), crop)

                if label == "washing":
                    saved_wash += 1
                else:
                    saved_not_wash += 1
        else:
            # Still need to feed bg subtractors even when not saving
            for i, (zone, bg_sub) in enumerate(zip(sink_zones, bg_subs)):
                crop = get_padded_crop(frame, zone)
                bg_sub.apply(crop)

        frame_idx += 1
        if frame_idx % (30 * 60) == 0:
            print(f"  {current_sec/60:.1f} min | wash: {saved_wash}, not_wash: {saved_not_wash}")

    cap.release()
    print(f"\nDone! Added to {out_dir}")
    print(f"  Washing:     {saved_wash}")
    print(f"  Not washing: {saved_not_wash}")


if __name__ == "__main__":
    main()
