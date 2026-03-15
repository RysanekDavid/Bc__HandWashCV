"""
AI-enhanced hand-wash event detector using MediaPipe Hands + motion analysis.

Combines two signals to reduce false positives:
  1. MOG2 background subtraction → motion in ROI  (same as baseline)
  2. MediaPipe Hands → are human hands actually present in ROI?

An event is recorded only when BOTH conditions are met:
  - Sufficient motion is detected (foreground pixels above threshold)
  - At least one hand is detected by MediaPipe in the ROI area

This eliminates false positives caused by shadows, water reflections,
or other non-hand movement that fooled the baseline detector.

Usage:
    python src/mediapipe_detector.py <video> <roi_json> [--csv out.csv] [--overlay out.mp4] [--no-preview]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from config import DetectionParams, OUTPUTS_DIR, PROJECT_ROOT
from roi_select import load_roi

# Path to the downloaded HandLandmarker model
_MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")


# ── Core detection ────────────────────────────────────────────

def detect_wash_events(
    video_path: str,
    roi: dict[str, int],
    params: DetectionParams = DetectionParams(),
    show_preview: bool = True,
    overlay_path: Optional[str] = None,
    hand_confidence: float = 0.5,
) -> pd.DataFrame:
    """
    Run combined motion + hand-detection wash detection on a single video.

    The detector requires BOTH motion in the ROI AND at least one hand
    detected by MediaPipe to count a frame as "active". This two-signal
    approach significantly reduces false positives compared to motion-only
    baseline.

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

    # Background subtractor (same as baseline)
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=params.bg_history,
        varThreshold=params.bg_var_threshold,
        detectShadows=False,
    )

    # MediaPipe HandLandmarker (Tasks API)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=hand_confidence,
        min_tracking_confidence=0.4,
    )
    hands = HandLandmarker.create_from_options(options)

    on_frames = int(params.wash_sec_on * fps)
    off_frames = int(params.wash_sec_off * fps)

    # Grace period: how many frames without hand detection before
    # we consider hands truly absent. MediaPipe can lose tracking
    # for a few frames even when hands are clearly visible, especially
    # with wet / shiny hands under industrial lighting.
    hand_grace_frames = int(3.0 * fps)  # 3 seconds

    # State machine (identical to baseline, with hand grace period)
    active = False
    move_cnt = 0
    still_cnt = 0
    hand_missing_cnt = 0
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

        # Signal 1: Motion (background subtraction)
        fg_mask = bg_sub.apply(roi_crop)
        fg_mask = cv2.medianBlur(fg_mask, params.median_blur_k)
        motion = int(np.count_nonzero(fg_mask))
        has_motion = motion > params.motion_thresh

        # Signal 2: Hand presence (MediaPipe HandLandmarker)
        roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = hands.detect_for_video(mp_image, timestamp_ms)
        has_hands_raw = len(result.hand_landmarks) > 0

        # Apply grace period: if hands were seen recently, still count as present
        if has_hands_raw:
            hand_missing_cnt = 0
        else:
            hand_missing_cnt += 1
        has_hands = hand_missing_cnt <= hand_grace_frames

        # Combined signal: both motion AND hands must be present
        frame_active = has_motion and has_hands

        # Update counters
        if frame_active:
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
        _draw_overlay(
            frame, x, y, w, h, active, motion, params.motion_thresh,
            has_hands, result.hand_landmarks, roi_crop.shape,
        )

        if show_preview:
            cv2.imshow("MediaPipe Hands Detector", frame)
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
    has_hands: bool,
    hand_landmarks,
    roi_shape: tuple,
) -> None:
    """Draw ROI rectangle, status, and hand landmarks on the frame."""
    color = (0, 0, 255) if active else (0, 255, 0)
    label = "WASHING" if active else "IDLE"
    hand_label = "HANDS" if has_hands else "NO HANDS"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        f"{label}  motion={motion}  thr={thresh}  [{hand_label}]",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Draw hand landmarks inside the ROI area (Tasks API uses NormalizedLandmark)
    if hand_landmarks:
        roi_h, roi_w = roi_shape[:2]
        for hand_lms in hand_landmarks:
            for lm in hand_lms:
                px = int(lm.x * roi_w) + x
                py = int(lm.y * roi_h) + y
                cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)


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
        description="AI-enhanced hand-wash detector (MediaPipe Hands + motion)."
    )
    parser.add_argument("video_path", help="Path to input video.")
    parser.add_argument("roi_json", help="Path to ROI JSON (from roi_select.py).")
    parser.add_argument("--csv", default=None, help="Output CSV path for detected events.")
    parser.add_argument("--overlay", default=None, help="Output overlay video path.")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window.")
    parser.add_argument("--hand-confidence", type=float, default=0.5,
                        help="MediaPipe hand detection confidence threshold (0-1).")
    parser.add_argument("--motion-thresh", type=int, default=None)
    parser.add_argument("--wash-on", type=float, default=None,
                        help="Seconds of activity to start event.")
    parser.add_argument("--wash-off", type=float, default=None,
                        help="Seconds of stillness to end event.")
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
        hand_confidence=args.hand_confidence,
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
