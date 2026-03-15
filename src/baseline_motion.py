"""
Baseline hand-wash event detector using background subtraction (MOG2).

Pipeline per frame:
  1. Crop to ROI (sink area)
  2. Apply MOG2 → foreground mask
  3. Denoise (median blur)
  4. Count foreground pixels  → "motion score"
  5. Hysteresis state machine:
       IDLE  → WASHING   when motion > thresh for  wash_sec_on  seconds
       WASHING → IDLE     when motion <= thresh for wash_sec_off seconds

Usage:
    python src/baseline_motion.py <video> <roi_json> [--csv out.csv] [--overlay out.mp4] [--no-preview]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from config import DetectionParams, OUTPUTS_DIR
from roi_select import load_roi


# ── Core detection ────────────────────────────────────────────

def detect_wash_events(
    video_path: str,
    roi: dict[str, int],
    params: DetectionParams = DetectionParams(),
    show_preview: bool = True,
    overlay_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run motion-based wash detection on a single video.

    Returns a DataFrame with columns:
        video, start_sec, end_sec, duration_sec
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=params.bg_history,
        varThreshold=params.bg_var_threshold,
        detectShadows=False,
    )

    on_frames = int(params.wash_sec_on * fps)
    off_frames = int(params.wash_sec_off * fps)

    # State machine
    active = False
    move_cnt = 0
    still_cnt = 0
    start_frame: Optional[int] = None
    frame_idx = 0
    events: list[dict] = []

    # Optional overlay writer
    vw = _create_writer(overlay_path, fps, width, height)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi_crop = frame[y : y + h, x : x + w]
        fg_mask = bg_sub.apply(roi_crop)
        fg_mask = cv2.medianBlur(fg_mask, params.median_blur_k)
        motion = int(np.count_nonzero(fg_mask))

        # Update counters
        if motion > params.motion_thresh:
            move_cnt += 1
            still_cnt = 0
        else:
            still_cnt += 1
            move_cnt = max(0, move_cnt - 1)

        # IDLE → WASHING
        if not active and move_cnt >= on_frames:
            active = True
            start_frame = frame_idx

        # WASHING → IDLE
        if active and still_cnt >= off_frames:
            events.append(_make_event(video_path, start_frame, frame_idx, fps))
            active = False
            start_frame = None
            move_cnt = 0
            still_cnt = 0

        # Draw overlay
        _draw_overlay(frame, x, y, w, h, active, motion, params.motion_thresh)

        if show_preview:
            cv2.imshow("Baseline Motion", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if vw is not None:
            vw.write(frame)

        frame_idx += 1

    # Edge case: video ends while a wash event is still active
    if active and start_frame is not None:
        events.append(_make_event(video_path, start_frame, frame_idx - 1, fps))

    _cleanup(cap, vw, show_preview)
    return pd.DataFrame(events)


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


def _create_writer(
    path: Optional[str], fps: float, width: int, height: int
) -> Optional[cv2.VideoWriter]:
    if path is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def _draw_overlay(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    active: bool, motion: int, thresh: int,
) -> None:
    color = (0, 0, 255) if active else (0, 255, 0)
    label = "WASHING" if active else "IDLE"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        f"{label}  motion={motion}  thr={thresh}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def _cleanup(
    cap: cv2.VideoCapture,
    vw: Optional[cv2.VideoWriter],
    had_preview: bool,
) -> None:
    cap.release()
    if vw is not None:
        vw.release()
    if had_preview:
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline motion-based hand-wash event detector."
    )
    parser.add_argument("video_path", help="Path to input video.")
    parser.add_argument("roi_json", help="Path to ROI JSON (from roi_select.py).")
    parser.add_argument("--csv", default=None, help="Output CSV path for detected events.")
    parser.add_argument("--overlay", default=None, help="Output overlay video path.")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window.")
    parser.add_argument("--motion-thresh", type=int, default=None)
    parser.add_argument("--wash-on", type=float, default=None, help="Seconds of motion to start event.")
    parser.add_argument("--wash-off", type=float, default=None, help="Seconds of stillness to end event.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roi = load_roi(Path(args.roi_json))

    params = DetectionParams()
    if args.motion_thresh is not None:
        params.motion_thresh = args.motion_thresh
    if args.wash_on is not None:
        params.wash_sec_on = args.wash_on
    if args.wash_off is not None:
        params.wash_sec_off = args.wash_off

    df = detect_wash_events(
        video_path=args.video_path,
        roi=roi,
        params=params,
        show_preview=not args.no_preview,
        overlay_path=args.overlay,
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
