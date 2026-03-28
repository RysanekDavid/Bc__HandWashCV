"""
YOLO classification-based hand-wash detector.

Runs a trained YOLO-cls model on padded sink-zone crops (per station)
with temporal smoothing and state-machine event detection.

The cropping logic matches generate_yolo_dataset.py exactly so that
inference sees the same field of view the model was trained on.

Usage:
    python src/yolo_cls_detector.py <video> [--model best.pt] [--no-preview]
"""

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, TRAINING_DIR, PROJECT_ROOT, DEFAULT_ROI_PATH

# Must match generate_yolo_dataset.py
PAD_X, PAD_Y = 150, 100


# ── Parameters ────────────────────────────────────────────────

@dataclass
class YoloCLSParams:
    """Tuneable parameters for the YOLO-cls detector."""

    confidence_threshold: float = 0.5
    """Smoothed washing probability above this → frame classified as washing."""

    smooth_window_sec: float = 0.5
    """Rolling-average window for temporal smoothing of predictions."""

    min_on_sec: float = 1.0
    """Consecutive washing-frames required to START an event."""

    min_off_sec: float = 2.0
    """Consecutive not-washing-frames required to END an event."""

    min_event_duration_sec: float = 3.0
    """Discard detected events shorter than this."""

    merge_gap_sec: float = 3.0
    """Merge events separated by less than this (same station only)."""


# ── Crop helper ───────────────────────────────────────────────

def get_padded_crop(frame: np.ndarray, zone: dict) -> np.ndarray:
    """Crop a sink zone with padding — identical to generate_yolo_dataset.py."""
    h, w = frame.shape[:2]
    zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
    x1 = max(0, zx - PAD_X)
    y1 = max(0, zy - PAD_Y)
    x2 = min(w, zx + zw + PAD_X)
    y2 = min(h, zy + zh + PAD_Y)
    return frame[y1:y2, x1:x2]


# ── Main detection function ──────────────────────────────────

def detect_wash_events(
    video_path: str,
    roi: dict[str, int],
    sink_zones: list[dict[str, int]],
    model_path: str,
    params: YoloCLSParams = YoloCLSParams(),
    show_preview: bool = True,
    imgsz: int = 224,
) -> pd.DataFrame:
    """
    Detect hand-wash events using YOLO image classification on sink-zone crops.

    Returns DataFrame with columns: video, start_sec, end_sec, duration_sec, station
    """
    from ultralytics import YOLO

    # Load YOLO model
    model = YOLO(model_path, verbose=False)
    print(f"Loaded YOLO model: {model_path}")

    # Determine class index for 'washing'
    class_names = model.names  # e.g. {0: 'not_washing', 1: 'washing'}
    washing_idx = None
    for idx, name in class_names.items():
        if name == "washing":
            washing_idx = idx
            break
    if washing_idx is None:
        raise ValueError(f"Model has no 'washing' class. Classes: {class_names}")
    print(f"Class mapping: {class_names}, washing_idx={washing_idx}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps
    print(f"Video: {fps:.1f} fps, {total_frames} frames, {total_sec/60:.1f} min")

    # Derived frame counts
    smooth_window = max(1, int(params.smooth_window_sec * fps))
    on_frames = max(1, int(params.min_on_sec * fps))
    off_frames = max(1, int(params.min_off_sec * fps))

    n_stations = len(sink_zones)

    # Per-station rolling buffers & state
    prob_buffers: list[deque] = [deque(maxlen=smooth_window) for _ in range(n_stations)]
    washing_cnt = [0] * n_stations   # consecutive frames classified as washing
    idle_cnt = [0] * n_stations      # consecutive frames classified as not washing
    is_washing = [False] * n_stations
    start_frames = [None] * n_stations

    all_events: list[dict] = []

    frame_idx = 0
    report_interval = int(fps * 60)  # report every ~1 minute

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for si, zone in enumerate(sink_zones):
            crop = get_padded_crop(frame, zone)

            # YOLO inference (silent)
            results = model.predict(crop, imgsz=imgsz, verbose=False)
            probs = results[0].probs

            # Get washing probability
            wash_prob = float(probs.data[washing_idx])
            prob_buffers[si].append(wash_prob)

            # Smoothed probability
            smoothed = sum(prob_buffers[si]) / len(prob_buffers[si])

            # Classification
            is_wash_frame = smoothed > params.confidence_threshold

            if not is_washing[si]:
                # IDLE state
                if is_wash_frame:
                    washing_cnt[si] += 1
                    idle_cnt[si] = 0
                else:
                    washing_cnt[si] = 0
                    idle_cnt[si] += 1

                if washing_cnt[si] >= on_frames:
                    is_washing[si] = True
                    start_frames[si] = frame_idx - on_frames + 1
                    idle_cnt[si] = 0
            else:
                # WASHING state
                if is_wash_frame:
                    idle_cnt[si] = 0
                else:
                    idle_cnt[si] += 1

                if idle_cnt[si] >= off_frames:
                    # End event
                    end_frame = frame_idx - off_frames
                    ev = _make_event(video_path, start_frames[si], end_frame, fps)
                    ev["station"] = si
                    all_events.append(ev)
                    is_washing[si] = False
                    start_frames[si] = None
                    washing_cnt[si] = 0

        # Preview
        if show_preview:
            vis = frame.copy()
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for si, zone in enumerate(sink_zones):
                zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
                color = (0, 0, 255) if is_washing[si] else (200, 200, 200)
                cv2.rectangle(vis, (zx, zy), (zx + zw, zy + zh), color, 2)

                if prob_buffers[si]:
                    prob = sum(prob_buffers[si]) / len(prob_buffers[si])
                    label = f"S{si+1}: {'WASH' if is_washing[si] else 'idle'} ({prob:.2f})"
                    cv2.putText(vis, label, (zx, zy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("YOLO-cls Detector", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        if frame_idx % report_interval == 0:
            current_sec = frame_idx / fps
            print(f"  {current_sec/60:.1f} min | events so far: {len(all_events)}")

    # Edge case: video ends during active washing
    for si in range(n_stations):
        if is_washing[si] and start_frames[si] is not None:
            ev = _make_event(video_path, start_frames[si], frame_idx - 1, fps)
            ev["station"] = si
            all_events.append(ev)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Sort by start time
    all_events.sort(key=lambda e: e["start_sec"])

    # Post-processing: merge close events (same station)
    if params.merge_gap_sec > 0 and len(all_events) > 1:
        merged = [all_events[0]]
        for ev in all_events[1:]:
            prev = merged[-1]
            gap = ev["start_sec"] - prev["end_sec"]
            if gap < params.merge_gap_sec and ev.get("station") == prev.get("station"):
                prev["end_sec"] = ev["end_sec"]
                prev["duration_sec"] = round(prev["end_sec"] - prev["start_sec"], 2)
            else:
                merged.append(ev)
        all_events = merged

    # Post-filter: minimum duration
    if params.min_event_duration_sec > 0:
        all_events = [e for e in all_events if e["duration_sec"] >= params.min_event_duration_sec]

    print(f"\nTotal events detected: {len(all_events)}")
    return pd.DataFrame(all_events)


# ── Helpers ───────────────────────────────────────────────────

def _make_event(video_path: str, start_frame: int, end_frame: int, fps: float) -> dict:
    start_sec = start_frame / fps
    end_sec = end_frame / fps
    return {
        "video": Path(video_path).name,
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "duration_sec": round(end_sec - start_sec, 2),
    }


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO-cls hand-wash detector on padded sink-zone crops."
    )
    parser.add_argument("video_path", help="Path to input video.")
    parser.add_argument("--model", default=str(TRAINING_DIR / "yolo26n_run" / "weights" / "best.pt"),
                        help="Path to trained YOLO-cls weights.")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH), help="Path to ROI JSON.")
    parser.add_argument("--csv", default=None, help="Output CSV path.")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Washing confidence threshold (default 0.5).")
    parser.add_argument("--smooth-window", type=float, default=0.5,
                        help="Temporal smoothing window in seconds (default 0.5).")
    parser.add_argument("--min-on", type=float, default=1.0,
                        help="Consecutive washing seconds to start event (default 1.0).")
    parser.add_argument("--min-off", type=float, default=2.0,
                        help="Consecutive idle seconds to end event (default 2.0).")
    parser.add_argument("--min-duration", type=float, default=3.0,
                        help="Minimum event duration in seconds (default 3.0).")
    parser.add_argument("--merge-gap", type=float, default=3.0,
                        help="Merge events closer than this (seconds, default 3.0).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roi_data = json.load(open(args.roi, encoding="utf-8"))
    sink_zones = roi_data.pop("sink_zones", None)
    if not sink_zones:
        print("ERROR: ROI JSON does not contain 'sink_zones'.")
        sys.exit(1)

    # Strip extra keys to get plain ROI
    roi_data.pop("soap_zones", None)
    roi_data.pop("soap_zone", None)
    roi = roi_data

    params = YoloCLSParams(
        confidence_threshold=args.confidence,
        smooth_window_sec=args.smooth_window,
        min_on_sec=args.min_on,
        min_off_sec=args.min_off,
        min_event_duration_sec=args.min_duration,
        merge_gap_sec=args.merge_gap,
    )

    df = detect_wash_events(
        video_path=args.video_path,
        roi=roi,
        sink_zones=sink_zones,
        model_path=args.model,
        params=params,
        show_preview=not args.no_preview,
    )

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False, encoding="utf-8")
        print(f"Events CSV saved: {args.csv}")

    if df.empty:
        print("No wash events detected.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
