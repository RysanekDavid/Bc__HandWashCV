"""
Interactive ROI selector.

Opens the first frame of a video and lets the user draw a rectangle
around the sink / hand-wash area.  The result is saved as JSON so
downstream scripts can crop to that region.

Usage:
    python src/roi_select.py <video_path> <out_roi_json>
    python src/roi_select.py                              # uses defaults from config
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


def select_roi(video_path: str) -> dict[str, int]:
    """Open first frame and let the user draw a rectangle. Returns dict(x, y, w, h)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    window_name = "Select ROI (drag rectangle, then press ENTER / SPACE)"
    rect = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled or invalid (width/height is 0).")

    return {"x": x, "y": y, "w": w, "h": h}


def save_roi(roi: dict[str, int], out_path: Path) -> None:
    """Persist ROI dict to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(roi, indent=2, ensure_ascii=False), encoding="utf-8")


def load_roi(roi_path: Path) -> dict[str, int]:
    """Load ROI dict from a JSON file."""
    with open(roi_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    from config import DATA_CLIPS_DIR, DEFAULT_ROI_PATH

    first_clip = sorted(DATA_CLIPS_DIR.glob("clip_*.mp4"))
    default_video = str(first_clip[0]) if first_clip else None

    parser = argparse.ArgumentParser(description="Interactive ROI selector for hand-wash detection.")
    parser.add_argument("video_path", nargs="?", default=default_video,
                        help="Path to a sample video (default: first clip in data_clips/processed/).")
    parser.add_argument("out_roi_json", nargs="?", default=str(DEFAULT_ROI_PATH),
                        help="Output JSON path (default: outputs/roi.json).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.video_path is None:
        print("ERROR: No video found. Pass a video path or place clips in data_clips/processed/.")
        sys.exit(1)

    print(f"Video : {args.video_path}")
    print(f"Output: {args.out_roi_json}")

    roi = select_roi(args.video_path)
    save_roi(roi, Path(args.out_roi_json))

    print(f"ROI saved: {roi}")


if __name__ == "__main__":
    main()
