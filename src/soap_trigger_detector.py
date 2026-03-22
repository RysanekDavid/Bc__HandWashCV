"""
AI hand-wash detector with soap-dispenser trigger zone.

Supports per-station multi-sink tracking: when sink_zones are defined
in roi.json, each station (soap zone + sink zone pair) gets an
independent state machine for concurrent hand-wash detection.

Detection logic per station:
  1. Hand detected by MediaPipe near the station's soap dispenser → TRIGGER start
  2. Motion continues in the station's sink zone                  → event ongoing
  3. Motion drops OR no hand in sink zone for timeout             → END event

Fallback: if no sink_zones defined, uses legacy global-ROI behaviour.

Usage:
    python src/soap_trigger_detector.py <video> <roi_json> [--csv out.csv] [--no-preview]
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from config import DetectionParams, OUTPUTS_DIR, PROJECT_ROOT
from roi_select import load_roi

_MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")


# ── Per-station state machine ────────────────────────────────

@dataclass
class StationTracker:
    """Independent state machine for one wash station (soap + sink pair)."""
    station_id: int
    soap_zone: dict        # {x, y, w, h} in absolute frame coords
    sink_zone: dict        # {x, y, w, h} in absolute frame coords (or None for global)

    # State
    washing: bool = False
    pending: bool = False
    still_cnt: int = 0
    soap_contact_cnt: int = 0
    pending_cnt: int = 0
    sink_frames: int = 0
    hand_missing_cnt: int = 0     # frames since last hand in sink zone
    start_frame: Optional[int] = None

    # Results
    events: list = field(default_factory=list)
    sink_frames_per_event: list = field(default_factory=list)


def _hand_landmarks_in_zone(hand_landmarks: list, zone: dict, roi: dict) -> bool:
    """Check if any hand landmark falls within a specific zone (absolute coords)."""
    if not hand_landmarks:
        return False

    roi_x, roi_y = roi["x"], roi["y"]
    roi_w, roi_h = roi["w"], roi["h"]
    zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]

    for hand_lms in hand_landmarks:
        for lm in hand_lms:
            abs_x = int(lm.x * roi_w) + roi_x
            abs_y = int(lm.y * roi_h) + roi_y
            if zx <= abs_x <= zx + zw and zy <= abs_y <= zy + zh:
                return True
    return False


def _hand_in_zone(hand_landmarks: list, zones: list[dict], roi: dict) -> bool:
    """Check if any hand landmark falls within ANY of the given zones."""
    for zone in zones:
        if _hand_landmarks_in_zone(hand_landmarks, zone, roi):
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


def _compute_zone_motion(fg_mask: np.ndarray, zone: dict, roi: dict) -> int:
    """Compute foreground pixel count within a specific zone's crop."""
    # Convert zone absolute coords to relative within ROI
    rel_x = max(0, zone["x"] - roi["x"])
    rel_y = max(0, zone["y"] - roi["y"])
    rel_x2 = min(roi["w"], rel_x + zone["w"])
    rel_y2 = min(roi["h"], rel_y + zone["h"])

    if rel_x2 <= rel_x or rel_y2 <= rel_y:
        return 0

    zone_mask = fg_mask[rel_y:rel_y2, rel_x:rel_x2]
    return int(np.count_nonzero(zone_mask))


def detect_wash_events(
    video_path: str,
    roi: dict[str, int],
    soap_zones: list[dict[str, int]],
    params: DetectionParams = DetectionParams(),
    show_preview: bool = True,
    overlay_path: Optional[str] = None,
    hand_confidence: float = 0.5,
    sink_zones: Optional[list[dict[str, int]]] = None,
) -> pd.DataFrame:
    """
    Detect hand-wash events using soap dispenser trigger + motion tracking.

    If sink_zones is provided and matches soap_zones in count, uses per-station
    tracking with independent state machines. Otherwise falls back to global ROI.

    Returns DataFrame with columns: video, start_sec, end_sec, duration_sec[, station]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Decide mode: per-station or legacy global
    use_stations = (
        sink_zones is not None
        and len(sink_zones) > 0
        and len(sink_zones) == len(soap_zones)
    )

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
        num_hands=4,  # support 2 people × 2 hands
        min_hand_detection_confidence=hand_confidence,
        min_tracking_confidence=0.4,
    )
    hands = HandLandmarker.create_from_options(options)

    off_frames = int(params.wash_sec_off * fps)
    min_contact_frames = max(1, int(params.soap_trigger_min_contact_sec * fps))
    confirm_frames = int(params.soap_post_trigger_confirm_sec * fps)
    use_pending_confirmation = confirm_frames > 0
    grace_frames = int(params.hand_detection_grace_sec * fps)
    hand_timeout_frames = int(params.station_hand_timeout_sec * fps)

    # Create station trackers or legacy single tracker
    if use_stations:
        stations = [
            StationTracker(station_id=i, soap_zone=sz, sink_zone=sk)
            for i, (sz, sk) in enumerate(zip(soap_zones, sink_zones))
        ]
    else:
        # Legacy: one station with global ROI as sink zone
        stations = [
            StationTracker(
                station_id=0,
                soap_zone={"x": 0, "y": 0, "w": 0, "h": 0},  # dummy, handled below
                sink_zone={"x": x, "y": y, "w": w, "h": h},  # full ROI
            )
        ]

    frame_idx = 0
    vw = _create_writer(overlay_path, fps, width, height)

    # Per-station grace counters (sink hand detection)
    sink_missing_per_station = [0] * len(stations)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi_crop = frame[y : y + h, x : x + w]

        # Global motion detection (for overlay / legacy fallback)
        fg_mask = bg_sub.apply(roi_crop)
        fg_mask = cv2.medianBlur(fg_mask, params.median_blur_k)
        ignore_top = int(h * params.soap_motion_ignore_top_ratio)
        motion_mask = fg_mask[ignore_top:, :] if ignore_top < h else fg_mask
        global_motion = int(np.count_nonzero(motion_mask))

        # Hand detection via MediaPipe
        roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = hands.detect_for_video(mp_image, timestamp_ms)
        hand_lms = result.hand_landmarks

        # === Per-station processing ===
        any_washing = False
        any_pending = False
        any_hand_at_soap = False

        for si, st in enumerate(stations):
            # --- Soap trigger for this station ---
            if use_stations:
                hand_at_soap = _hand_landmarks_in_zone(hand_lms, st.soap_zone, roi)
            else:
                # Legacy mode: check all soap zones
                hand_at_soap = _hand_in_zone(hand_lms, soap_zones, roi)

            if hand_at_soap:
                any_hand_at_soap = True

            # --- Motion in this station's sink zone ---
            if use_stations:
                zone_motion = _compute_zone_motion(fg_mask, st.sink_zone, roi)
                # Scale threshold proportionally to zone size vs full ROI
                zone_area = st.sink_zone["w"] * st.sink_zone["h"]
                roi_area = w * h
                zone_thresh = max(100, int(params.motion_thresh * zone_area / roi_area))
                has_motion = zone_motion > zone_thresh
            else:
                has_motion = global_motion > params.motion_thresh

            # --- Hand in this station's sink zone (with grace period) ---
            if use_stations:
                hand_in_sink_raw = _hand_landmarks_in_zone(hand_lms, st.sink_zone, roi)
            else:
                hand_in_sink_raw = _hand_in_sink_band(hand_lms, roi, params.soap_sink_min_y_ratio)

            if hand_in_sink_raw:
                sink_missing_per_station[si] = 0
            else:
                sink_missing_per_station[si] += 1
            hand_in_sink = sink_missing_per_station[si] <= grace_frames

            # --- State machine ---
            if not st.washing and not st.pending:
                # IDLE
                if hand_at_soap:
                    st.soap_contact_cnt += 1
                else:
                    st.soap_contact_cnt = 0

                if st.soap_contact_cnt >= min_contact_frames:
                    st.start_frame = frame_idx - min_contact_frames + 1
                    if use_pending_confirmation:
                        st.pending = True
                        st.pending_cnt = 0
                    else:
                        st.washing = True
                        st.still_cnt = 0
                        st.hand_missing_cnt = 0
                        st.soap_contact_cnt = 0

            elif st.pending:
                st.pending_cnt += 1
                if has_motion and hand_in_sink:
                    st.washing = True
                    st.pending = False
                    st.still_cnt = 0
                    st.hand_missing_cnt = 0
                    st.soap_contact_cnt = 0
                    st.pending_cnt = 0
                elif st.pending_cnt >= confirm_frames:
                    st.pending = False
                    st.start_frame = None
                    st.pending_cnt = 0
                    st.soap_contact_cnt = 0

            else:
                # WASHING
                if hand_in_sink:
                    st.sink_frames += 1
                    st.hand_missing_cnt = 0

                    # Also reset still_cnt if hands are present (per-station)
                    if use_stations:
                        st.still_cnt = 0
                else:
                    st.hand_missing_cnt += 1

                # Legacy mode: also track motion
                if has_motion:
                    st.still_cnt = 0
                else:
                    st.still_cnt += 1

                # End conditions depend on mode:
                if use_stations:
                    # Per-station: end when no HAND in sink zone for timeout
                    # (hand landmarker is the authority, not pixel motion)
                    should_end = st.hand_missing_cnt >= hand_timeout_frames
                else:
                    # Legacy: end when no MOTION for wash_sec_off
                    should_end = st.still_cnt >= off_frames

                if should_end:
                    ev = _make_event(video_path, st.start_frame, frame_idx, fps)
                    ev["station"] = st.station_id
                    st.events.append(ev)
                    st.sink_frames_per_event.append(st.sink_frames)
                    st.washing = False
                    st.start_frame = None
                    st.still_cnt = 0
                    st.soap_contact_cnt = 0
                    st.sink_frames = 0
                    st.hand_missing_cnt = 0

            if st.washing:
                any_washing = True
            if st.pending:
                any_pending = True

        # Draw overlay
        hand_in_sink_global = _hand_in_sink_band(hand_lms, roi, params.soap_sink_min_y_ratio)
        _draw_overlay(frame, roi, soap_zones, any_washing, any_pending, global_motion,
                      params.motion_thresh, len(hand_lms) > 0, any_hand_at_soap,
                      hand_in_sink_global, hand_lms,
                      sink_zones if use_stations else None, stations if use_stations else None)

        if show_preview:
            cv2.imshow("Soap Trigger Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if vw is not None:
            vw.write(frame)

        frame_idx += 1

    # Edge case: video ends during active events
    for st in stations:
        if st.washing and st.start_frame is not None:
            ev = _make_event(video_path, st.start_frame, frame_idx - 1, fps)
            ev["station"] = st.station_id
            st.events.append(ev)
            st.sink_frames_per_event.append(st.sink_frames)

    _cleanup(cap, vw, show_preview)

    # Collect all events from all stations
    all_events = []
    all_sink_frames = []
    for st in stations:
        all_events.extend(st.events)
        all_sink_frames.extend(st.sink_frames_per_event)

    # Sort by start time
    if all_events:
        paired = sorted(zip(all_events, all_sink_frames), key=lambda p: p[0]["start_sec"])
        all_events, all_sink_frames = zip(*paired)
        all_events = list(all_events)
        all_sink_frames = list(all_sink_frames)

    # Post-processing: merge close events (only within same station)
    merge_gap = params.merge_gap_sec
    if merge_gap > 0 and len(all_events) > 1:
        merged = [all_events[0]]
        merged_sink = [all_sink_frames[0]]
        for ev, sf in zip(all_events[1:], all_sink_frames[1:]):
            prev = merged[-1]
            gap = ev["start_sec"] - prev["end_sec"]
            same_station = ev.get("station") == prev.get("station")
            if gap < merge_gap and same_station:
                prev["end_sec"] = ev["end_sec"]
                prev["duration_sec"] = round(prev["end_sec"] - prev["start_sec"], 2)
                merged_sink[-1] += sf
            else:
                merged.append(ev)
                merged_sink.append(sf)
        all_events = merged
        all_sink_frames = merged_sink

    # Post-filter: duration and sink-time
    min_dur = params.soap_min_event_duration_sec
    min_sink = params.soap_min_sink_time_sec
    if (min_dur > 0 or min_sink > 0) and all_events:
        filtered = []
        for ev, sf in zip(all_events, all_sink_frames):
            if min_dur > 0 and ev["duration_sec"] < min_dur:
                continue
            if min_sink > 0 and sf / fps < min_sink:
                continue
            filtered.append(ev)
        all_events = filtered

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
    sink_zones: Optional[list[dict]] = None,
    stations: Optional[list] = None,
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

    # Sink zones (if per-station)
    if sink_zones:
        for i, sk in enumerate(sink_zones):
            skx, sky, skw, skh = sk["x"], sk["y"], sk["w"], sk["h"]
            # Color based on station state
            if stations and i < len(stations):
                st = stations[i]
                if st.washing:
                    sk_color = (0, 0, 255)  # red = washing
                elif st.pending:
                    sk_color = (0, 165, 255)  # orange = pending
                else:
                    sk_color = (200, 200, 200)  # gray = idle
            else:
                sk_color = (200, 200, 200)
            cv2.rectangle(frame, (skx, sky), (skx + skw, sky + skh), sk_color, 2)
            cv2.putText(frame, f"SINK{i+1}", (skx, sky - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sk_color, 1)

    # Status text
    label = "WASHING" if washing else ("PENDING" if pending else "IDLE")
    soap_label = "SOAP!" if hand_at_soap else ""
    sink_label = "SINK" if hand_in_sink else ""
    hand_label = "HANDS" if has_hands else ""

    # Per-station status
    if stations and len(stations) > 1:
        station_info = "  ".join(
            f"S{st.station_id+1}:{'WASH' if st.washing else 'idle'}"
            for st in stations
        )
        info_text = f"{label}  motion={motion}  [{hand_label}]  {soap_label} {sink_label}  |  {station_info}"
    else:
        info_text = f"{label}  motion={motion}  [{hand_label}]  {soap_label} {sink_label}"

    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

    # Sink zones for per-station tracking
    sink_zones = roi_data.pop("sink_zones", None)

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
        sink_zones=sink_zones,
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
