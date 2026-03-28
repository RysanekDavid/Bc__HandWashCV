"""
AI hand-wash detector with soap-dispenser trigger zone.

Detection logic (3 signals):
  1. Hand detected by MediaPipe near the soap dispenser → TRIGGER start
  2. Motion continues in the main ROI (sink area)        → event ongoing
  3. Motion drops below threshold for wash_sec_off sec   → END event

This approach solves two key problems:
  - Precise event start: triggered by physical contact with soap dispenser
  - Eliminates false positives: passing through the frame without touching
    soap does not trigger an event

Usage:
    python src/soap_trigger_detector.py <video> <roi_json> [--csv out.csv] [--no-preview]
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

_MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")


def _hand_in_zone(hand_landmarks: list, zones: list[dict], roi: dict) -> bool:
    """
    Check if any detected hand landmark falls within any of the soap zones.

    hand_landmarks: list of NormalizedLandmark lists from MediaPipe
    zones: list of soap zone dicts [{x, y, w, h}, ...] in absolute frame coords
    roi: main ROI dict {x, y, w, h} — landmarks are relative to this crop
    """
    if not hand_landmarks:
        return False

    roi_x, roi_y = roi["x"], roi["y"]
    roi_w, roi_h = roi["w"], roi["h"]

    for hand_lms in hand_landmarks:
        for lm in hand_lms:
            abs_x = int(lm.x * roi_w) + roi_x
            abs_y = int(lm.y * roi_h) + roi_y

            for zone in zones:
                zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
                if zx <= abs_x <= zx + zw and zy <= abs_y <= zy + zh:
                    return True
    return False


def _hand_in_sink_band(hand_landmarks: list, roi: dict, min_y_ratio: float) -> bool:
    """Return True if any hand landmark is in the lower sink band of ROI."""
    if not hand_landmarks:
        return False

    roi_h = roi["h"]
    y_threshold = int(min_y_ratio * roi_h)

    for hand_lms in hand_landmarks:
        for lm in hand_lms:
            rel_y = int(lm.y * roi_h)
            if rel_y >= y_threshold:
                return True
    return False


def detect_wash_events(
    video_path: str,
    roi: dict[str, int],
    soap_zones: list[dict[str, int]],
    params: DetectionParams = DetectionParams(),
    show_preview: bool = True,
    overlay_path: Optional[str] = None,
    hand_confidence: float = 0.5,
) -> pd.DataFrame:
    """
    Detect hand-wash events using soap dispenser trigger + motion tracking.

    Flow per event:
      1. IDLE: wait for hand detected in any soap zone
      2. TRIGGERED → WASHING: soap contact detected, start event
      3. WASHING: track motion in main ROI
      4. Motion gone for wash_sec_off seconds → END event, back to IDLE

    Returns DataFrame with columns: video, start_sec, end_sec, duration_sec
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=params.bg_history,
        varThreshold=params.bg_var_threshold,
        detectShadows=False,
    )

    # MediaPipe HandLandmarker
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

    off_frames = int(params.wash_sec_off * fps)
    min_contact_frames = max(1, int(params.soap_trigger_min_contact_sec * fps))
    confirm_frames = int(params.soap_post_trigger_confirm_sec * fps)
    use_pending_confirmation = confirm_frames > 0
    grace_frames = int(params.hand_detection_grace_sec * fps)

    # State machine
    washing = False
    pending = False
    still_cnt = 0
    soap_contact_cnt = 0
    pending_cnt = 0
    sink_frames = 0
    soap_missing_cnt = 0    # grace period counter for soap zone
    sink_missing_cnt = 0    # grace period counter for sink band
    start_frame: Optional[int] = None
    frame_idx = 0
    events: list[dict] = []
    sink_frames_per_event: list[int] = []

    vw = _create_writer(overlay_path, fps, width, height)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi_crop = frame[y : y + h, x : x + w]

        # Motion detection
        fg_mask = bg_sub.apply(roi_crop)
        fg_mask = cv2.medianBlur(fg_mask, params.median_blur_k)
        ignore_top = int(h * params.soap_motion_ignore_top_ratio)
        motion_mask = fg_mask[ignore_top:, :] if ignore_top < h else fg_mask
        motion = int(np.count_nonzero(motion_mask))
        has_motion = motion > params.motion_thresh

        # Hand detection via MediaPipe
        roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = hands.detect_for_video(mp_image, timestamp_ms)
        hand_lms = result.hand_landmarks

        # Check if hand touches any soap zone (with grace period)
        hand_at_soap_raw = _hand_in_zone(hand_lms, soap_zones, roi)
        hand_in_sink_raw = _hand_in_sink_band(hand_lms, roi, params.soap_sink_min_y_ratio)

        # Apply grace period: tolerate brief detection dropouts
        if hand_at_soap_raw:
            soap_missing_cnt = 0
        else:
            soap_missing_cnt += 1
        hand_at_soap = soap_missing_cnt <= grace_frames

        if hand_in_sink_raw:
            sink_missing_cnt = 0
        else:
            sink_missing_cnt += 1
        hand_in_sink = sink_missing_cnt <= grace_frames

        # --- State machine ---

        if not washing and not pending:
            # IDLE: require a short, continuous soap-zone contact to avoid tap noise.
            if hand_at_soap:
                soap_contact_cnt += 1
            else:
                soap_contact_cnt = 0

            if soap_contact_cnt >= min_contact_frames:
                start_frame = frame_idx - min_contact_frames + 1
                if use_pending_confirmation:
                    pending = True
                    pending_cnt = 0
                else:
                    washing = True
                    still_cnt = 0
                    soap_contact_cnt = 0

        elif pending:
            pending_cnt += 1
            # PENDING → WASHING: confirm sink interaction shortly after trigger.
            if has_motion and hand_in_sink:
                washing = True
                pending = False
                still_cnt = 0
                soap_contact_cnt = 0
                pending_cnt = 0
            elif pending_cnt >= confirm_frames:
                # Cancel accidental trigger if sink confirmation never arrived.
                pending = False
                start_frame = None
                pending_cnt = 0
                soap_contact_cnt = 0
        else:
            # WASHING: track motion to determine end
            if has_motion:
                still_cnt = 0
            else:
                still_cnt += 1

            if hand_in_sink:
                sink_frames += 1

            # WASHING → IDLE: no motion for wash_sec_off seconds
            if still_cnt >= off_frames:
                events.append(_make_event(video_path, start_frame, frame_idx, fps))
                sink_frames_per_event.append(sink_frames)
                washing = False
                start_frame = None
                still_cnt = 0
                soap_contact_cnt = 0
                sink_frames = 0

        # Draw overlay
        _draw_overlay(frame, roi, soap_zones, washing, pending, motion,
                  params.motion_thresh, len(hand_lms) > 0, hand_at_soap,
                  hand_in_sink, hand_lms)

        if show_preview:
            cv2.imshow("Soap Trigger Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if vw is not None:
            vw.write(frame)

        frame_idx += 1

    # Edge case: video ends during active event
    if washing and start_frame is not None:
        events.append(_make_event(video_path, start_frame, frame_idx - 1, fps))
        sink_frames_per_event.append(sink_frames)

    _cleanup(cap, vw, show_preview)

    # Post-processing: merge close events, then apply filters
    merge_gap = params.merge_gap_sec
    if merge_gap > 0 and len(events) > 1:
        merged = [events[0]]
        merged_sink = [sink_frames_per_event[0]]
        for ev, sf in zip(events[1:], sink_frames_per_event[1:]):
            prev = merged[-1]
            gap = ev["start_sec"] - prev["end_sec"]
            if gap < merge_gap:
                # Merge: extend previous event
                prev["end_sec"] = ev["end_sec"]
                prev["duration_sec"] = round(prev["end_sec"] - prev["start_sec"], 2)
                merged_sink[-1] += sf
            else:
                merged.append(ev)
                merged_sink.append(sf)
        events = merged
        sink_frames_per_event = merged_sink

    # Post-filter: apply both duration and sink-time filters
    min_dur = params.soap_min_event_duration_sec
    min_sink = params.soap_min_sink_time_sec
    if (min_dur > 0 or min_sink > 0) and events:
        filtered = []
        for ev, sf in zip(events, sink_frames_per_event):
            if min_dur > 0 and ev["duration_sec"] < min_dur:
                continue
            if min_sink > 0 and sf / fps < min_sink:
                continue
            filtered.append(ev)
        events = filtered

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
    roi: dict, soap_zones: list[dict],
    washing: bool, pending: bool, motion: int, thresh: int,
    has_hands: bool, hand_at_soap: bool,
    hand_in_sink: bool,
    hand_landmarks,
) -> None:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Main ROI
    roi_color = (0, 0, 255) if washing else ((0, 165, 255) if pending else (0, 255, 0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)

    # All soap zones
    for i, sz in enumerate(soap_zones):
        sx, sy, sw, sh = sz["x"], sz["y"], sz["w"], sz["h"]
        soap_color = (0, 255, 255) if hand_at_soap else (255, 255, 0)
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), soap_color, 2)
        cv2.putText(frame, f"SOAP{i+1}", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, soap_color, 1)

    # Status text
    label = "WASHING" if washing else ("PENDING" if pending else "IDLE")
    soap_label = "SOAP!" if hand_at_soap else ""
    sink_label = "SINK" if hand_in_sink else ""
    hand_label = "HANDS" if has_hands else ""
    cv2.putText(
        frame,
        f"{label}  motion={motion}  [{hand_label}]  {soap_label} {sink_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Hand landmarks
    if hand_landmarks:
        roi_h, roi_w = h, w
        for hand_lms in hand_landmarks:
            for lm in hand_lms:
                px = int(lm.x * roi_w) + x
                py = int(lm.y * roi_h) + y
                cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)


def _cleanup(cap, vw, had_preview):
    cap.release()
    if vw is not None:
        vw.release()
    if had_preview:
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand-wash detector with soap dispenser trigger zone."
    )
    parser.add_argument("video_path", help="Path to input video.")
    parser.add_argument("roi_json", help="Path to ROI JSON (must contain 'soap_zone' key).")
    parser.add_argument("--csv", default=None, help="Output CSV path.")
    parser.add_argument("--overlay", default=None, help="Output overlay video path.")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--motion-thresh", type=int, default=None)
    parser.add_argument("--wash-off", type=float, default=None)
    parser.add_argument("--soap-min-contact", type=float, default=None,
                        help="Minimum continuous soap-zone contact in seconds (anti-FP).")
    parser.add_argument("--soap-confirm-window", type=float, default=None,
                        help="Post-trigger confirmation window in seconds (0 disables pending confirmation).")
    parser.add_argument("--soap-sink-min-y", type=float, default=None,
                        help="Lower sink band threshold as ROI Y ratio (0..1).")
    parser.add_argument("--soap-ignore-top", type=float, default=None,
                        help="Top ROI ratio ignored for motion counting (0..1).")
    parser.add_argument("--soap-min-duration", type=float, default=None,
                        help="Discard events shorter than this many seconds.")
    parser.add_argument("--soap-min-sink-time", type=float, default=None,
                        help="Minimum cumulative hand-in-sink seconds during event.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roi_data = load_roi(Path(args.roi_json))

    # Support both 'soap_zones' (list) and legacy 'soap_zone' (single dict)
    soap_zones = roi_data.pop("soap_zones", None)
    if soap_zones is None:
        single = roi_data.pop("soap_zone", None)
        if single is None:
            print("ERROR: ROI JSON does not contain 'soap_zones'.")
            print("Run: python src/roi_select.py --soap-zone")
            sys.exit(1)
        soap_zones = [single]
    else:
        roi_data.pop("soap_zone", None)

    roi = roi_data

    params = DetectionParams()
    if args.motion_thresh is not None:
        params.motion_thresh = args.motion_thresh
    if args.wash_off is not None:
        params.wash_sec_off = args.wash_off
    if args.soap_min_contact is not None:
        params.soap_trigger_min_contact_sec = max(0.0, args.soap_min_contact)
    if args.soap_confirm_window is not None:
        params.soap_post_trigger_confirm_sec = max(0.0, args.soap_confirm_window)
    if args.soap_sink_min_y is not None:
        params.soap_sink_min_y_ratio = min(max(args.soap_sink_min_y, 0.0), 1.0)
    if args.soap_ignore_top is not None:
        params.soap_motion_ignore_top_ratio = min(max(args.soap_ignore_top, 0.0), 1.0)
    if hasattr(args, 'soap_min_duration') and args.soap_min_duration is not None:
        params.soap_min_event_duration_sec = max(0.0, args.soap_min_duration)
    if hasattr(args, 'soap_min_sink_time') and args.soap_min_sink_time is not None:
        params.soap_min_sink_time_sec = max(0.0, args.soap_min_sink_time)

    df = detect_wash_events(
        video_path=args.video_path,
        roi=roi,
        soap_zones=soap_zones,
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
