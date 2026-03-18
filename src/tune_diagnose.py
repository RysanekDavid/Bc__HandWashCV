"""
Diagnostic tuning tool for the soap-trigger detector.

Runs the detector on a small set of clips and reports detailed
state-machine transitions so you can see exactly why events are
detected or missed.

Usage:
    python src/tune_diagnose.py clip_0105.mp4 clip_0123.mp4 clip_0128.mp4
    python src/tune_diagnose.py clip_0105.mp4 --soap-min-contact 0.15 --soap-min-duration 4
    python src/tune_diagnose.py --all-errors          # auto-pick clips with TP/FP/FN issues

The report shows for each clip:
  - Every trigger fire (hand touched soap zone)
  - State transitions: IDLE → PENDING → WASHING → IDLE
  - Raw event start/end times and durations
  - Which events were filtered (too short, pending cancelled)
  - Comparison with Ground Truth (TP / FP / FN)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from config import DetectionParams, LABELED_DIR, OUTPUTS_DIR, PROJECT_ROOT
from roi_select import load_roi

_MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")


def _hand_in_zone(hand_landmarks, zones, roi):
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


def _hand_in_sink_band(hand_landmarks, roi, min_y_ratio):
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


# ── Diagnostic data structures ────────────────────────────────

class DiagEvent:
    """One raw event captured by the state machine (before post-filters)."""
    def __init__(self, trigger_sec, start_sec, end_sec, sink_time_sec=0.0, filtered_reason=None):
        self.trigger_sec = trigger_sec   # when soap zone contact began
        self.start_sec = start_sec       # event start (trigger armed)
        self.end_sec = end_sec           # event end (motion stopped)
        self.duration = round(end_sec - start_sec, 2)
        self.sink_time_sec = round(sink_time_sec, 2)
        self.filtered_reason = filtered_reason  # None = kept, else reason string

    @property
    def kept(self):
        return self.filtered_reason is None


class DiagTransition:
    """A single state-machine transition."""
    def __init__(self, sec, from_state, to_state, detail=""):
        self.sec = sec
        self.from_state = from_state
        self.to_state = to_state
        self.detail = detail


def diagnose_clip(
    video_path: str,
    roi: dict,
    soap_zones: list[dict],
    params: DetectionParams,
    hand_confidence: float = 0.5,
):
    """Run detector with full diagnostics. Returns (events, transitions, stats)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=params.bg_history,
        varThreshold=params.bg_var_threshold,
        detectShadows=False,
    )

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

    # State
    washing = False
    pending = False
    still_cnt = 0
    soap_contact_cnt = 0
    pending_cnt = 0
    sink_frames_cnt = 0
    start_frame: Optional[int] = None
    trigger_frame: Optional[int] = None
    frame_idx = 0

    raw_events: list[DiagEvent] = []
    transitions: list[DiagTransition] = []

    # Stats
    total_soap_contacts = 0  # how many frames hand was in soap zone
    total_hand_frames = 0    # how many frames hand was detected at all
    total_motion_frames = 0  # how many frames had motion above threshold
    max_motion = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi_crop = frame[y : y + h, x : x + w]
        sec = frame_idx / fps

        # Motion
        fg_mask = bg_sub.apply(roi_crop)
        fg_mask = cv2.medianBlur(fg_mask, params.median_blur_k)
        ignore_top = int(h * params.soap_motion_ignore_top_ratio)
        motion_mask = fg_mask[ignore_top:, :] if ignore_top < h else fg_mask
        motion = int(np.count_nonzero(motion_mask))
        has_motion = motion > params.motion_thresh
        if has_motion:
            total_motion_frames += 1
        max_motion = max(max_motion, motion)

        # Hands
        roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = hands.detect_for_video(mp_image, timestamp_ms)
        hand_lms = result.hand_landmarks
        has_hands = len(hand_lms) > 0
        if has_hands:
            total_hand_frames += 1

        hand_at_soap = _hand_in_zone(hand_lms, soap_zones, roi)
        hand_in_sink = _hand_in_sink_band(hand_lms, roi, params.soap_sink_min_y_ratio)
        if hand_at_soap:
            total_soap_contacts += 1

        # --- State machine (identical to detector, but with logging) ---

        if not washing and not pending:
            if hand_at_soap:
                soap_contact_cnt += 1
            else:
                soap_contact_cnt = 0

            if soap_contact_cnt >= min_contact_frames:
                trigger_frame = frame_idx - min_contact_frames + 1
                start_frame = trigger_frame
                if use_pending_confirmation:
                    pending = True
                    pending_cnt = 0
                    transitions.append(DiagTransition(
                        sec, "IDLE", "PENDING",
                        f"soap contact {soap_contact_cnt} frames "
                        f"(need {min_contact_frames})"
                    ))
                else:
                    washing = True
                    still_cnt = 0
                    transitions.append(DiagTransition(
                        sec, "IDLE", "WASHING",
                        f"soap contact {soap_contact_cnt} frames "
                        f"(need {min_contact_frames})"
                    ))
                    soap_contact_cnt = 0

        elif pending:
            pending_cnt += 1
            if has_motion and hand_in_sink:
                washing = True
                pending = False
                still_cnt = 0
                transitions.append(DiagTransition(
                    sec, "PENDING", "WASHING",
                    f"confirmed: motion={motion}, hand_in_sink=True "
                    f"after {pending_cnt} frames"
                ))
                soap_contact_cnt = 0
                pending_cnt = 0
            elif pending_cnt >= confirm_frames:
                transitions.append(DiagTransition(
                    sec, "PENDING", "IDLE (cancelled)",
                    f"no confirmation in {confirm_frames} frames "
                    f"({params.soap_post_trigger_confirm_sec}s)"
                ))
                pending = False
                start_frame = None
                trigger_frame = None
                pending_cnt = 0
                soap_contact_cnt = 0
        else:
            if has_motion:
                still_cnt = 0
            else:
                still_cnt += 1

            if hand_in_sink:
                sink_frames_cnt += 1

            if still_cnt >= off_frames:
                end_sec = frame_idx / fps
                start_sec = start_frame / fps
                raw_events.append(DiagEvent(
                    trigger_sec=trigger_frame / fps if trigger_frame else start_sec,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    sink_time_sec=sink_frames_cnt / fps,
                ))
                transitions.append(DiagTransition(
                    sec, "WASHING", "IDLE",
                    f"no motion for {params.wash_sec_off}s → "
                    f"event {start_sec:.1f}–{end_sec:.1f}s "
                    f"(dur={end_sec - start_sec:.1f}s, sink={sink_frames_cnt / fps:.1f}s)"
                ))
                washing = False
                start_frame = None
                trigger_frame = None
                still_cnt = 0
                soap_contact_cnt = 0

        frame_idx += 1

    # Edge case: video ends during active event
    if washing and start_frame is not None:
        end_sec = (frame_idx - 1) / fps
        start_sec = start_frame / fps
        raw_events.append(DiagEvent(
            trigger_sec=trigger_frame / fps if trigger_frame else start_sec,
            start_sec=start_sec,
            end_sec=end_sec,
            sink_time_sec=sink_frames_cnt / fps,
        ))
        transitions.append(DiagTransition(
            end_sec, "WASHING", "IDLE (video end)",
            f"event {start_sec:.1f}–{end_sec:.1f}s (dur={end_sec - start_sec:.1f}s, sink={sink_frames_cnt / fps:.1f}s)"
        ))

    cap.release()

    # Post-filter: mark short events and low sink-time events
    min_dur = params.soap_min_event_duration_sec
    min_sink = params.soap_min_sink_time_sec
    if min_dur > 0:
        for ev in raw_events:
            if ev.duration < min_dur:
                ev.filtered_reason = f"too short ({ev.duration}s < {min_dur}s)"
    if min_sink > 0:
        for ev in raw_events:
            if ev.filtered_reason is None and ev.sink_time_sec < min_sink:
                ev.filtered_reason = f"low sink time ({ev.sink_time_sec}s < {min_sink}s)"

    total_frames = frame_idx
    stats = {
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 1),
        "fps": fps,
        "hand_frames": total_hand_frames,
        "hand_pct": round(100 * total_hand_frames / max(total_frames, 1), 1),
        "soap_contact_frames": total_soap_contacts,
        "soap_contact_pct": round(100 * total_soap_contacts / max(total_frames, 1), 1),
        "motion_frames": total_motion_frames,
        "motion_pct": round(100 * total_motion_frames / max(total_frames, 1), 1),
        "max_motion": max_motion,
    }

    return raw_events, transitions, stats


# ── IoU matching (same as evaluate.py) ────────────────────────

def _calculate_iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0


def _match_events(gt_events, pred_events, iou_threshold=0.1):
    """Match GT vs predicted. Returns (tp, fp, fn, matches)."""
    matched_gt = set()
    matched_pred = set()
    matches = []

    for i, gt in enumerate(gt_events):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(pred_events):
            if j in matched_pred:
                continue
            iou = _calculate_iou(gt["start_sec"], gt["end_sec"],
                                 pred["start_sec"], pred["end_sec"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            matched_gt.add(i)
            matched_pred.add(best_j)
            matches.append((i, best_j, round(best_iou, 3)))

    tp = len(matches)
    fn = len(gt_events) - tp
    fp = len(pred_events) - tp
    return tp, fp, fn, matches


# ── Pretty printing ──────────────────────────────────────────

def print_clip_report(clip_name, raw_events, transitions, stats, gt_events):
    """Print a detailed diagnostic report for one clip."""
    kept_events = [e for e in raw_events if e.kept]
    filtered_events = [e for e in raw_events if not e.kept]

    print(f"\n{'='*70}")
    print(f"  {clip_name}")
    print(f"{'='*70}")

    # Stats overview
    print(f"  Duration: {stats['duration_sec']}s  |  FPS: {stats['fps']}")
    print(f"  Hand detected: {stats['hand_pct']}% of frames  |  "
          f"Soap contact: {stats['soap_contact_frames']} frames ({stats['soap_contact_pct']}%)")
    print(f"  Motion: {stats['motion_pct']}% of frames  |  Max motion: {stats['max_motion']}")

    # GT summary
    if gt_events:
        gt_str = ", ".join(f"{e['start_sec']:.1f}–{e['end_sec']:.1f}s" for e in gt_events)
        print(f"  GT events ({len(gt_events)}): {gt_str}")
    else:
        print(f"  GT events: none (should be empty)")

    # State transitions timeline
    print(f"\n  Timeline (state transitions):")
    if not transitions:
        print(f"    (no transitions — detector stayed IDLE entire clip)")
    for t in transitions:
        print(f"    {t.sec:6.1f}s  {t.from_state:20s} → {t.to_state}")
        if t.detail:
            print(f"             {t.detail}")

    # Raw events
    print(f"\n  Raw events (before post-filter): {len(raw_events)}")
    for i, ev in enumerate(raw_events):
        status = "✓ KEPT" if ev.kept else f"✗ FILTERED: {ev.filtered_reason}"
        print(f"    [{i+1}] trigger={ev.trigger_sec:.1f}s  "
              f"event={ev.start_sec:.1f}–{ev.end_sec:.1f}s  "
              f"dur={ev.duration}s  sink={ev.sink_time_sec}s  {status}")

    # Final kept events
    print(f"\n  Final predictions: {len(kept_events)}")

    # GT comparison
    if gt_events or kept_events:
        pred_dicts = [{"start_sec": e.start_sec, "end_sec": e.end_sec} for e in kept_events]
        tp, fp, fn, matches = _match_events(gt_events, pred_dicts)

        for gi, pi, iou in matches:
            gt = gt_events[gi]
            pr = pred_dicts[pi]
            print(f"    TP: GT[{gt['start_sec']:.1f}–{gt['end_sec']:.1f}] "
                  f"↔ Pred[{pr['start_sec']:.1f}–{pr['end_sec']:.1f}]  IoU={iou}")

        matched_pred_idx = {pi for _, pi, _ in matches}
        matched_gt_idx = {gi for gi, _, _ in matches}
        for j, pr in enumerate(pred_dicts):
            if j not in matched_pred_idx:
                print(f"    FP: Pred[{pr['start_sec']:.1f}–{pr['end_sec']:.1f}]  "
                      f"(no matching GT)")
        for i, gt in enumerate(gt_events):
            if i not in matched_gt_idx:
                print(f"    FN: GT[{gt['start_sec']:.1f}–{gt['end_sec']:.1f}]  "
                      f"(no matching prediction)")

        verdict = "OK" if fp == 0 and fn == 0 else ""
        if fp > 0:
            verdict += f" {fp}FP"
        if fn > 0:
            verdict += f" {fn}FN"
        print(f"    → TP={tp}  FP={fp}  FN={fn}  {verdict.strip()}")
    else:
        print(f"    → TN (correct: no GT, no predictions)")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tuning tool for soap-trigger detector. "
                    "Give clip names (e.g. clip_0105.mp4) to analyze."
    )
    parser.add_argument("clips", nargs="*",
                        help="Clip filenames (e.g. clip_0105.mp4). Searched in data_clips/labeled/.")
    parser.add_argument("--all-errors", action="store_true",
                        help="Auto-select all clips that had TP, FP, or FN in last evaluation.")
    parser.add_argument("--all-gt", action="store_true",
                        help="Auto-select all clips that have GT events.")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--motion-thresh", type=int, default=None)
    parser.add_argument("--wash-off", type=float, default=None)
    parser.add_argument("--soap-min-contact", type=float, default=None)
    parser.add_argument("--soap-confirm-window", type=float, default=None)
    parser.add_argument("--soap-sink-min-y", type=float, default=None)
    parser.add_argument("--soap-ignore-top", type=float, default=None)
    parser.add_argument("--soap-min-duration", type=float, default=None)
    parser.add_argument("--soap-min-sink-time", type=float, default=None)
    args = parser.parse_args()

    # Build params
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
    if args.soap_min_duration is not None:
        params.soap_min_event_duration_sec = max(0.0, args.soap_min_duration)
    if args.soap_min_sink_time is not None:
        params.soap_min_sink_time_sec = max(0.0, args.soap_min_sink_time)

    # Load ROI + soap zones
    roi_data = load_roi(OUTPUTS_DIR / "roi.json")
    soap_zones = roi_data.pop("soap_zones", None)
    if soap_zones is None:
        single = roi_data.pop("soap_zone", None)
        if single is None:
            print("ERROR: No soap_zones in roi.json. Run: python src/roi_select.py --soap-zone")
            sys.exit(1)
        soap_zones = [single]
    else:
        roi_data.pop("soap_zone", None)
    roi = roi_data

    # Load GT
    annotations_path = OUTPUTS_DIR / "annotations.json"
    with open(annotations_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Determine which clips to analyze
    clip_names = []
    if args.all_errors:
        # Pick clips from last eval that had any FP or FN
        eval_csv = OUTPUTS_DIR / "eval_soap_trigger.csv"
        if eval_csv.exists():
            df = pd.read_csv(eval_csv)
            for _, row in df.iterrows():
                gt_n = row["gt_count"]
                pred_n = row["pred_count"]
                match_n = row["match"]
                has_fp = pred_n > match_n
                has_fn = gt_n > match_n
                if has_fp or has_fn:
                    clip_names.append(row["video"])
        else:
            print("No eval CSV found. Run evaluate.py first or specify clips manually.")
            sys.exit(1)
    elif args.all_gt:
        for name, data in gt_data.items():
            if data.get("events"):
                clip_names.append(name)
        clip_names.sort()
    elif args.clips:
        for c in args.clips:
            if not c.endswith(".mp4"):
                c = c + ".mp4"
            clip_names.append(c)
    else:
        print("Specify clip names, or use --all-errors / --all-gt")
        parser.print_help()
        sys.exit(1)

    # Print params header
    print(f"{'='*70}")
    print(f"  DIAGNOSTIC TUNING REPORT")
    print(f"{'='*70}")
    print(f"  Clips: {len(clip_names)}")
    print(f"  Parameters:")
    print(f"    motion_thresh       = {params.motion_thresh}")
    print(f"    wash_sec_off        = {params.wash_sec_off}")
    print(f"    min_contact         = {params.soap_trigger_min_contact_sec}s")
    print(f"    confirm_window      = {params.soap_post_trigger_confirm_sec}s")
    print(f"    sink_min_y_ratio    = {params.soap_sink_min_y_ratio}")
    print(f"    ignore_top_ratio    = {params.soap_motion_ignore_top_ratio}")
    print(f"    min_event_duration  = {params.soap_min_event_duration_sec}s")
    print(f"    min_sink_time       = {params.soap_min_sink_time_sec}s")

    # Run diagnostics
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for clip_name in clip_names:
        clip_path = LABELED_DIR / clip_name
        if not clip_path.exists():
            print(f"\n  SKIP: {clip_name} not found in {LABELED_DIR}")
            continue

        gt_events = gt_data.get(clip_name, {}).get("events", [])

        raw_events, transitions, stats = diagnose_clip(
            video_path=str(clip_path),
            roi=roi,
            soap_zones=soap_zones,
            params=params,
            hand_confidence=args.hand_confidence,
        )

        print_clip_report(clip_name, raw_events, transitions, stats, gt_events)

        # Accumulate totals
        kept = [e for e in raw_events if e.kept]
        pred_dicts = [{"start_sec": e.start_sec, "end_sec": e.end_sec} for e in kept]
        if gt_events or pred_dicts:
            tp, fp, fn, _ = _match_events(gt_events, pred_dicts)
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
        else:
            totals["tn"] += 1

    # Summary
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{'='*70}")
    print(f"  TOTALS ({len(clip_names)} clips)")
    print(f"{'='*70}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={totals['tn']}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
