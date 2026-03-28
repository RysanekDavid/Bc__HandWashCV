"""
Visual debug tool for hand-wash detection pipeline.

Shows a comprehensive diagnostic overlay on video playback:
  - Main ROI (green/red)
  - Soap trigger zones (yellow/cyan)
  - Hand landmarks (magenta dots)
  - Motion heatmap (foreground mask)
  - Real-time signal bars (motion level, hand confidence)
  - State machine status (IDLE/WASHING)
  - Ground Truth events overlay (if annotations available)
  - Frame-by-frame stepping and parameter adjustment

Controls:
  SPACE    - Pause / Resume
  D        - Step forward 1 frame (while paused)
  A        - Step backward ~1 sec (while paused)
  Q / ESC  - Quit
  M        - Toggle motion heatmap overlay
  H        - Toggle hand landmarks
  G        - Toggle ground truth bar
  +/-      - Adjust motion_thresh live (±500)
  S        - Save current frame as PNG

Usage:
    python src/debug_viewer.py <video> <roi_json>
    python src/debug_viewer.py <video> <roi_json> --detector soap_trigger
    python src/debug_viewer.py <video> <roi_json> --detector baseline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import DetectionParams, OUTPUTS_DIR, GT_DIR, PROJECT_ROOT
from roi_select import load_roi

_MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")


def _load_gt(video_name: str) -> list[dict]:
    """Load ground truth events for a clip if available."""
    ann_path = GT_DIR / "annotations.json"
    if not ann_path.exists():
        return []
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(video_name, {}).get("events", [])


def _in_gt(sec: float, gt_events: list[dict]) -> bool:
    """Check if current timestamp falls within any GT event."""
    return any(e["start_sec"] <= sec <= e["end_sec"] for e in gt_events)


def _hand_in_zones(hand_landmarks, zones, roi):
    """Check if any hand landmark is in any soap zone."""
    if not hand_landmarks or not zones:
        return False
    roi_x, roi_y, roi_w, roi_h = roi["x"], roi["y"], roi["w"], roi["h"]
    for hand_lms in hand_landmarks:
        for lm in hand_lms:
            ax = int(lm.x * roi_w) + roi_x
            ay = int(lm.y * roi_h) + roi_y
            for z in zones:
                if z["x"] <= ax <= z["x"]+z["w"] and z["y"] <= ay <= z["y"]+z["h"]:
                    return True
    return False


def _hand_in_sink_band(hand_landmarks, roi, min_y_ratio):
    """Check if any hand landmark is in lower sink band of ROI."""
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


def run_debug(
    video_path: str,
    roi: dict,
    soap_zones: list[dict],
    params: DetectionParams,
    detector_name: str,
    hand_confidence: float = 0.5,
):
    """Main debug viewer loop."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = Path(video_path).name
    gt_events = _load_gt(video_name)

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=params.bg_history,
        varThreshold=params.bg_var_threshold,
        detectShadows=False,
    )

    # MediaPipe Hands (for mediapipe/soap_trigger modes)
    hands = None
    if detector_name in ("mediapipe", "soap_trigger"):
        BaseOptions = mp.tasks.BaseOptions
        HL = mp.tasks.vision.HandLandmarker
        HLO = mp.tasks.vision.HandLandmarkerOptions
        options = HLO(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=hand_confidence,
            min_tracking_confidence=0.4,
        )
        hands = HL.create_from_options(options)

    # Display toggles
    show_heatmap = False
    show_hands = True
    show_gt = True
    paused = False

    # State machine
    active = False
    pending = False
    move_cnt = 0
    still_cnt = 0
    hand_missing_cnt = 0
    soap_contact_cnt = 0
    pending_cnt = 0
    start_frame = None
    events = []

    on_frames = int(params.wash_sec_on * fps)
    off_frames = int(params.wash_sec_off * fps)
    min_contact_frames = max(1, int(params.soap_trigger_min_contact_sec * fps))
    confirm_frames = int(params.soap_post_trigger_confirm_sec * fps)
    use_pending_confirmation = confirm_frames > 0
    hand_grace = int(3.0 * fps)
    motion_thresh = params.motion_thresh

    frame_idx = 0
    window_name = f"DEBUG: {video_name} [{detector_name}]"

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            # Re-use last frame while paused
            pass

        display = frame.copy()
        current_sec = frame_idx / fps

        # ── Signals ──
        roi_crop = frame[y:y+h, x:x+w]
        fg_mask = bg_sub.apply(roi_crop)
        fg_mask = cv2.medianBlur(fg_mask, params.median_blur_k)
        ignore_top = int(h * params.soap_motion_ignore_top_ratio)
        motion_mask = fg_mask[ignore_top:, :] if ignore_top < h else fg_mask
        motion = int(np.count_nonzero(motion_mask))
        has_motion = motion > motion_thresh

        hand_lms = []
        has_hands = False
        hand_at_soap = False
        if hands is not None:
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            ts_ms = int(frame_idx * 1000 / fps)
            result = hands.detect_for_video(mp_image, ts_ms)
            hand_lms = result.hand_landmarks
            has_hands = len(hand_lms) > 0
            if soap_zones:
                hand_at_soap = _hand_in_zones(hand_lms, soap_zones, roi)
        hand_in_sink = _hand_in_sink_band(hand_lms, roi, params.soap_sink_min_y_ratio)

        # ── State machine ──
        if detector_name == "baseline":
            frame_active = has_motion
            if frame_active:
                move_cnt += 1; still_cnt = 0
            else:
                still_cnt += 1; move_cnt = max(0, move_cnt - 1)
            if not active and move_cnt >= on_frames:
                active = True; start_frame = frame_idx
            if active and still_cnt >= off_frames:
                events.append((start_frame/fps, frame_idx/fps))
                active = False; start_frame = None; move_cnt = 0; still_cnt = 0

        elif detector_name == "mediapipe":
            if has_hands:
                hand_missing_cnt = 0
            else:
                hand_missing_cnt += 1
            hands_smooth = hand_missing_cnt <= hand_grace
            frame_active = has_motion and hands_smooth
            if frame_active:
                move_cnt += 1; still_cnt = 0
            else:
                still_cnt += 1; move_cnt = max(0, move_cnt - 1)
            if not active and move_cnt >= on_frames:
                active = True; start_frame = frame_idx
            if active and still_cnt >= off_frames:
                events.append((start_frame/fps, frame_idx/fps))
                active = False; start_frame = None; move_cnt = 0; still_cnt = 0

        elif detector_name == "soap_trigger":
            if not active and not pending:
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
                        active = True
                        still_cnt = 0
                        soap_contact_cnt = 0

            elif pending:
                pending_cnt += 1
                if has_motion and hand_in_sink:
                    active = True
                    pending = False
                    still_cnt = 0
                    soap_contact_cnt = 0
                    pending_cnt = 0
                elif pending_cnt >= confirm_frames:
                    pending = False
                    start_frame = None
                    pending_cnt = 0
                    soap_contact_cnt = 0

            if active:
                if has_motion:
                    still_cnt = 0
                else:
                    still_cnt += 1
                if still_cnt >= off_frames:
                    events.append((start_frame/fps, frame_idx/fps))
                    active = False; start_frame = None; still_cnt = 0
                    soap_contact_cnt = 0

        # ── Draw: Motion heatmap ──
        if show_heatmap:
            heatmap = cv2.applyColorMap(fg_mask, cv2.COLORMAP_JET)
            display[y:y+h, x:x+w] = cv2.addWeighted(roi_crop, 0.5, heatmap, 0.5, 0)

        # ── Draw: ROI box ──
        roi_color = (0, 0, 255) if active else ((0, 165, 255) if pending else (0, 255, 0))
        cv2.rectangle(display, (x, y), (x+w, y+h), roi_color, 2)

        # ── Draw: Soap zones ──
        for i, sz in enumerate(soap_zones):
            sc = (0, 255, 255) if hand_at_soap else (255, 255, 0)
            cv2.rectangle(display, (sz["x"], sz["y"]),
                          (sz["x"]+sz["w"], sz["y"]+sz["h"]), sc, 2)
            cv2.putText(display, f"SOAP{i+1}", (sz["x"], sz["y"]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)

        # ── Draw: Hand landmarks ──
        if show_hands and hand_lms:
            for hlm in hand_lms:
                for lm in hlm:
                    px = int(lm.x * w) + x
                    py = int(lm.y * h) + y
                    cv2.circle(display, (px, py), 3, (255, 0, 255), -1)

        # ── Draw: Info panel (top) ──
        state_label = "WASHING" if active else ("PENDING" if pending else "IDLE")
        hand_label = "HANDS" if has_hands else "NO HANDS"
        soap_label = " SOAP!" if hand_at_soap else ""
        sink_label = " SINK" if hand_in_sink else ""
        info = f"{state_label} | motion={motion}/{motion_thresh} | {hand_label}{soap_label}{sink_label}"
        cv2.putText(display, info, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        time_info = f"Frame {frame_idx}/{total_frames} | {current_sec:.1f}s / {total_frames/fps:.1f}s"
        cv2.putText(display, time_info, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        det_info = f"[{detector_name}] thresh={motion_thresh} wash_off={params.wash_sec_off}s"
        cv2.putText(display, det_info, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Draw: Signal bars (bottom-left) ──
        bar_y = height - 80
        # Motion bar
        motion_pct = min(motion / (motion_thresh * 3), 1.0)
        bar_w = int(200 * motion_pct)
        bar_color = (0, 0, 255) if has_motion else (0, 100, 0)
        cv2.rectangle(display, (20, bar_y), (20 + bar_w, bar_y + 20), bar_color, -1)
        cv2.rectangle(display, (20, bar_y), (220, bar_y + 20), (100, 100, 100), 1)
        thresh_x = 20 + int(200 * (1/3))  # thresh line at 1/3
        cv2.line(display, (thresh_x, bar_y), (thresh_x, bar_y + 20), (255, 255, 255), 2)
        cv2.putText(display, "Motion", (225, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Counter bar (move_cnt or still_cnt progress)
        if detector_name != "soap_trigger":
            if not active:
                prog = min(move_cnt / max(on_frames, 1), 1.0)
                cv2.rectangle(display, (20, bar_y + 25), (20 + int(200*prog), bar_y + 45), (0, 200, 200), -1)
                cv2.putText(display, f"ON: {move_cnt}/{on_frames}", (225, bar_y + 41),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                prog = min(still_cnt / max(off_frames, 1), 1.0)
                cv2.rectangle(display, (20, bar_y + 25), (20 + int(200*prog), bar_y + 45), (200, 100, 0), -1)
                cv2.putText(display, f"OFF: {still_cnt}/{off_frames}", (225, bar_y + 41),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            if active:
                prog = min(still_cnt / max(off_frames, 1), 1.0)
                cv2.rectangle(display, (20, bar_y + 25), (20 + int(200*prog), bar_y + 45), (200, 100, 0), -1)
                cv2.putText(display, f"OFF: {still_cnt}/{off_frames}", (225, bar_y + 41),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Draw: GT timeline bar (bottom) ──
        if show_gt and gt_events:
            tl_y = height - 25
            tl_w = width - 40
            cv2.rectangle(display, (20, tl_y), (20 + tl_w, tl_y + 15), (50, 50, 50), -1)
            total_sec = total_frames / fps
            for ev in gt_events:
                sx = 20 + int(ev["start_sec"] / total_sec * tl_w)
                ex = 20 + int(ev["end_sec"] / total_sec * tl_w)
                cv2.rectangle(display, (sx, tl_y), (ex, tl_y + 15), (0, 180, 0), -1)
            # Current position marker
            cur_x = 20 + int(current_sec / total_sec * tl_w)
            cv2.line(display, (cur_x, tl_y - 3), (cur_x, tl_y + 18), (0, 0, 255), 2)
            cv2.putText(display, "GT", (20 + tl_w + 5, tl_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

            # In GT indicator
            if _in_gt(current_sec, gt_events):
                cv2.putText(display, "IN GT", (width - 100, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Draw: Pause indicator ──
        if paused:
            cv2.putText(display, "PAUSED (D=step, A=back, SPACE=play)",
                        (width//2 - 200, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Draw: Controls help ──
        controls = "SPACE=pause D=step A=back M=heatmap H=hands G=gt +/-=thresh Q=quit S=save"
        cv2.putText(display, controls, (20, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        cv2.imshow(window_name, display)

        # ── Input handling ──
        wait_ms = 1 if not paused else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') and paused:
            # Step forward
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
        elif key == ord('a') and paused:
            # Step backward ~1 sec
            target = max(0, frame_idx - int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_idx = target
            # Re-init bg subtractor since we jumped
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=params.bg_history,
                varThreshold=params.bg_var_threshold,
                detectShadows=False,
            )
            ok, frame = cap.read()
        elif key == ord('m'):
            show_heatmap = not show_heatmap
        elif key == ord('h'):
            show_hands = not show_hands
        elif key == ord('g'):
            show_gt = not show_gt
        elif key == ord('+') or key == ord('='):
            motion_thresh += 500
            print(f"motion_thresh = {motion_thresh}")
        elif key == ord('-'):
            motion_thresh = max(500, motion_thresh - 500)
            print(f"motion_thresh = {motion_thresh}")
        elif key == ord('s'):
            save_path = OUTPUTS_DIR / f"debug_{video_name}_f{frame_idx}.png"
            cv2.imwrite(str(save_path), display)
            print(f"Saved: {save_path}")

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Print detected events summary
    if events:
        print(f"\nDetected {len(events)} event(s):")
        for i, (s, e) in enumerate(events):
            print(f"  Event {i+1}: {s:.2f}s – {e:.2f}s ({e-s:.1f}s)")
    else:
        print("\nNo events detected.")

    if gt_events:
        print(f"Ground Truth: {len(gt_events)} event(s)")
        for i, ev in enumerate(gt_events):
            print(f"  GT {i+1}: {ev['start_sec']:.2f}s – {ev['end_sec']:.2f}s")


# ── CLI ───────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Visual debug tool for hand-wash detection.")
    parser.add_argument("video_path", help="Path to video clip.")
    parser.add_argument("roi_json", help="Path to ROI JSON.")
    parser.add_argument("--detector", choices=["baseline", "mediapipe", "soap_trigger"],
                        default="soap_trigger", help="Detector logic to visualize (default: soap_trigger).")
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
    return parser.parse_args()


def main():
    args = _parse_args()

    roi_data = load_roi(Path(args.roi_json))
    soap_zones = roi_data.pop("soap_zones", None)
    if soap_zones is None:
        single = roi_data.pop("soap_zone", None)
        soap_zones = [single] if single else []
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
    if args.soap_min_duration is not None:
        params.soap_min_event_duration_sec = max(0.0, args.soap_min_duration)

    print(f"Video    : {args.video_path}")
    print(f"Detector : {args.detector}")
    print(f"ROI      : {roi}")
    print(f"Soap zones: {len(soap_zones)}")
    print(f"Params   : thresh={params.motion_thresh}, on={params.wash_sec_on}s, off={params.wash_sec_off}s")
    print()

    run_debug(
        video_path=args.video_path,
        roi=roi,
        soap_zones=soap_zones,
        params=params,
        detector_name=args.detector,
        hand_confidence=args.hand_confidence,
    )


if __name__ == "__main__":
    main()
