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


def select_roi(video_path: str, prompt: str = "Select ROI (drag rectangle, then press ENTER / SPACE)") -> dict[str, int]:
    """Open first frame and let the user draw a rectangle. Returns dict(x, y, w, h)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    rect = cv2.selectROI(prompt, frame, fromCenter=False, showCrosshair=True)
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
    parser.add_argument("--soap-zone", action="store_true",
                        help="Select soap dispenser zone(s). Repeat selection for each dispenser; press ESC/cancel to finish.")
    parser.add_argument("--sink-zone", action="store_true",
                        help="Select per-sink zone(s) for multi-station tracking. One zone per sink/umyvadlo.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.video_path is None:
        print("ERROR: No video found. Pass a video path or place clips in data_clips/processed/.")
        sys.exit(1)

    print(f"Video : {args.video_path}")
    print(f"Output: {args.out_roi_json}")

    out_path = Path(args.out_roi_json)

    # Load existing ROI if adding zones to existing file
    if (args.soap_zone or args.sink_zone) and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            roi_data = json.load(f)
        if args.soap_zone:
            roi_data.pop("soap_zone", None)
            roi_data.pop("soap_zones", None)
        if args.sink_zone:
            roi_data.pop("sink_zones", None)
        print(f"Existing ROI: x={roi_data['x']}, y={roi_data['y']}, w={roi_data['w']}, h={roi_data['h']}")
    else:
        print("Step 1: Select the SINK area (main ROI)")
        roi_data = select_roi(args.video_path, "Step 1: Select SINK area, then press ENTER")
        print(f"ROI saved: {roi_data}")

    if args.soap_zone:
        soap_zones = []
        i = 1
        while True:
            print(f"\nSelect SOAP DISPENSER #{i} (press ESC or draw empty rect to finish)")
            try:
                soap = select_roi(args.video_path, f"Select SOAP DISPENSER #{i}, then ENTER (ESC to finish)")
                soap_zones.append(soap)
                print(f"  Soap zone #{i}: {soap}")
                i += 1
            except RuntimeError:
                # User cancelled (ESC or empty rect) → done
                break

        if soap_zones:
            roi_data["soap_zones"] = soap_zones
            print(f"\n{len(soap_zones)} soap zone(s) saved.")
        else:
            print("\nNo soap zones selected.")

    if args.sink_zone:
        sink_zones = []
        i = 1
        while True:
            print(f"\nSelect SINK ZONE #{i} (area around umyvadlo #{i}, press ESC to finish)")
            try:
                sink = select_roi(args.video_path, f"Select SINK ZONE #{i}, then ENTER (ESC to finish)")
                sink_zones.append(sink)
                print(f"  Sink zone #{i}: {sink}")
                i += 1
            except RuntimeError:
                break

        if sink_zones:
            roi_data["sink_zones"] = sink_zones
            print(f"\n{len(sink_zones)} sink zone(s) saved.")
        else:
            print("\nNo sink zones selected.")

    save_roi(roi_data, out_path)
    print(f"\nAll saved to: {out_path}")


if __name__ == "__main__":
    main()
