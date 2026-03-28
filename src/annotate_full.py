"""
Full-video annotation tool for hand-wash events.

Instead of 20-second clips, this tool works directly on a full raw video.
You watch the video and mark wash events with S (start) and E (end).
Progress is auto-saved, so you can quit and resume where you left off.

Controls:
    SPACE       Pause / Resume
    S           Mark wash START
    E           Mark wash END
    U           Undo pending start
    D           Delete last event
    RIGHT (→)   Skip forward  5 seconds
    LEFT  (←)   Skip backward 5 seconds
    PAGE UP     Skip forward 30 seconds
    PAGE DOWN   Skip backward 30 seconds
    +/-         Speed up / slow down (0.5x–8x)
    Q           Save and quit
    ESC         Quit without saving

Output:  outputs/ground_truth/full_video_gt.json

Usage:
    python src/annotate_full.py data_clips/2026-02-06/20260127_193759_tp00002.mp4
    python src/annotate_full.py --resume   # resume previous session
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import GT_DIR, DEFAULT_ROI_PATH
from roi_select import load_roi


GT_PATH = GT_DIR / "full_video_gt.json"
WINDOW_NAME = "Full Video Annotator  (S=start E=end Q=save+quit)"
SPEED_OPTIONS = [0.5, 1.0, 2.0, 4.0, 8.0]


# ── Persistence ───────────────────────────────────────────────

def load_gt() -> dict:
    if GT_PATH.exists():
        with open(GT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_gt(data: dict) -> None:
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Formatting ────────────────────────────────────────────────

def _fmt(sec: float) -> str:
    """Format seconds as HH:MM:SS.d"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    if h > 0:
        return f"{h}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


# ── Drawing ───────────────────────────────────────────────────

def _draw_hud(
    frame, t: float, total: float, paused: bool, speed: float,
    events: list, pending_start, roi: dict | None, video_name: str,
):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Video name
    cv2.putText(frame, video_name, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # Time
    status = "⏸ PAUSED" if paused else f"▶ {speed:.1f}x"
    cv2.putText(frame, f"{_fmt(t)} / {_fmt(total)}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status, (320, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Event count
    cv2.putText(frame, f"Events: {len(events)}", (w - 180, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Progress bar
    bar_y = 75
    bar_x = 15
    bar_w = w - 30
    progress = t / total if total > 0 else 0

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (60, 60, 60), -1)

    # Event blocks on bar
    for ev in events:
        sx = bar_x + int((ev["start_sec"] / total) * bar_w)
        ex = bar_x + int((ev["end_sec"] / total) * bar_w)
        cv2.rectangle(frame, (sx, bar_y), (max(ex, sx + 2), bar_y + 12), (0, 200, 0), -1)

    # Playhead
    px = bar_x + int(progress * bar_w)
    cv2.rectangle(frame, (px - 1, bar_y - 2), (px + 1, bar_y + 14), (0, 255, 255), -1)

    # Pending start marker
    if pending_start is not None:
        mx = bar_x + int((pending_start / total) * bar_w)
        cv2.line(frame, (mx, bar_y - 4), (mx, bar_y + 16), (0, 0, 255), 2)
        cv2.putText(frame, f"START @ {_fmt(pending_start)}", (w - 300, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Recent events list (bottom-left)
    y_off = h - 15
    for i, ev in enumerate(reversed(events[-8:])):
        dur = ev["end_sec"] - ev["start_sec"]
        txt = f"#{len(events)-i}: {_fmt(ev['start_sec'])} -> {_fmt(ev['end_sec'])}  ({dur:.0f}s)"
        cv2.putText(frame, txt, (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        y_off -= 20

    # ROI
    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 0), 1)

    # Controls reminder
    cv2.putText(frame, "S=start E=end  D=del  <-/-> 5s  PgUp/Dn 30s  +/- speed  Q=save",
                (15, h - 5 - 20 * min(len(events), 8) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ── Core loop ─────────────────────────────────────────────────

def annotate_full(video_path: Path, roi, resume_sec: float = 0.0,
                  existing_events: list | None = None) -> tuple:
    """
    Returns (events, last_position_sec) or (None, 0) if ESC pressed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps
    video_name = video_path.name

    events = list(existing_events) if existing_events else []
    pending_start = None
    paused = True  # Start paused
    speed_idx = 1  # 1.0x

    # Resume position
    if resume_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(resume_sec * fps))
        print(f"  Resuming from {_fmt(resume_sec)}")

    print(f"\n  Video: {video_name} ({_fmt(total_sec)})")
    print(f"  Events loaded: {len(events)}")
    print(f"  Press SPACE to start playback, Q to save & quit\n")

    while True:
        t0 = time.perf_counter()

        if not paused:
            ok, frame = cap.read()
            if not ok:
                # End of video
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
                ok, frame = cap.read()
                if not ok:
                    break
                paused = True
                print("  Reached end of video — paused.")
        else:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
            ok, frame = cap.read()
            if not ok:
                break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_sec = current_frame / fps

        _draw_hud(frame, current_sec, total_sec, paused,
                  SPEED_OPTIONS[speed_idx], events, pending_start, roi, video_name)

        cv2.imshow(WINDOW_NAME, frame)

        target_delay = (1000 / fps) / SPEED_OPTIONS[speed_idx]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        wait_ms = max(1, int(target_delay - elapsed_ms))
        key = cv2.waitKeyEx(wait_ms if not paused else 0)

        # ── Keys ──
        # Windows arrow key codes (waitKeyEx returns full code)
        KEY_LEFT   = 2424832
        KEY_RIGHT  = 2555904
        KEY_UP     = 2490368
        KEY_DOWN   = 2621440
        KEY_PGUP   = 2162688
        KEY_PGDN   = 2228224

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None, current_sec

        elif key == ord("q") or key == ord("Q"):  # Save & quit
            if pending_start is not None:
                print(f"  WARNING: Discarded pending start @ {_fmt(pending_start)}")
            cap.release()
            cv2.destroyAllWindows()
            return events, current_sec

        elif key == ord(" "):
            paused = not paused

        elif key == ord("s") or key == ord("S"):
            pending_start = current_sec
            print(f"  START @ {_fmt(current_sec)}")

        elif key == ord("e") or key == ord("E"):
            if pending_start is None:
                print("  WARNING: Press S first!")
            elif current_sec <= pending_start:
                print("  WARNING: End must be after start")
            else:
                ev = {"start_sec": round(pending_start, 2),
                      "end_sec": round(current_sec, 2)}
                events.append(ev)
                dur = ev["end_sec"] - ev["start_sec"]
                print(f"  EVENT #{len(events)}: {_fmt(ev['start_sec'])} -> {_fmt(ev['end_sec'])} ({dur:.1f}s)")
                pending_start = None

        elif key == ord("u") or key == ord("U"):
            if pending_start is not None:
                print(f"  Cleared pending start @ {_fmt(pending_start)}")
                pending_start = None

        elif key == ord("d") or key == ord("D"):
            if events:
                removed = events.pop()
                print(f"  Deleted event: {_fmt(removed['start_sec'])} -> {_fmt(removed['end_sec'])}")

        elif key == ord("+") or key == ord("="):
            speed_idx = min(speed_idx + 1, len(SPEED_OPTIONS) - 1)
            print(f"  Speed: {SPEED_OPTIONS[speed_idx]}x")

        elif key == ord("-"):
            speed_idx = max(speed_idx - 1, 0)
            print(f"  Speed: {SPEED_OPTIONS[speed_idx]}x")

        elif key in (KEY_RIGHT, ord("l")):  # → +5s
            new_f = min(current_frame + int(5 * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)

        elif key in (KEY_LEFT, ord("h")):  # ← -5s
            new_f = max(current_frame - int(5 * fps), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)

        elif key in (KEY_PGUP, KEY_UP):  # PgUp / ↑ → +30s
            new_f = min(current_frame + int(30 * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)

        elif key in (KEY_PGDN, KEY_DOWN):  # PgDn / ↓ → -30s
            new_f = max(current_frame - int(30 * fps), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)

    cap.release()
    cv2.destroyAllWindows()
    return events, total_sec


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full-video annotation tool.")
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved position")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH))
    args = parser.parse_args()

    # Load ROI
    roi = None
    roi_path = Path(args.roi)
    if roi_path.exists():
        roi = load_roi(roi_path)

    # Load existing GT
    gt_data = load_gt()

    # Resolve video
    if args.video:
        video_path = Path(args.video)
    elif gt_data.get("video_path"):
        video_path = Path(gt_data["video_path"])
    else:
        print("ERROR: Provide a video path or use --resume")
        sys.exit(1)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Resume state
    existing_events = gt_data.get("events", []) if str(video_path) == gt_data.get("video_path") else []
    resume_sec = gt_data.get("last_position_sec", 0.0) if args.resume and existing_events else 0.0

    print(f"Full-Video Annotator")
    print(f"  Video: {video_path}")
    print(f"  Output: {GT_PATH}")

    events, last_pos = annotate_full(video_path, roi, resume_sec, existing_events)

    if events is not None:
        # Sort events by start time
        events.sort(key=lambda e: e["start_sec"])

        gt_data = {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "last_position_sec": round(last_pos, 2),
            "events": events,
        }
        save_gt(gt_data)
        print(f"\n  Saved {len(events)} events to {GT_PATH}")
        print(f"  Last position: {_fmt(last_pos)}")

        # Quick summary
        print(f"\n  === Events ===")
        for i, ev in enumerate(events, 1):
            dur = ev["end_sec"] - ev["start_sec"]
            print(f"    {i:2d}. {_fmt(ev['start_sec'])} -> {_fmt(ev['end_sec'])}  ({dur:.1f}s)")
    else:
        # ESC — save position anyway for resume
        if "video_path" not in gt_data:
            gt_data["video_path"] = str(video_path)
            gt_data["video_name"] = video_path.name
        gt_data["last_position_sec"] = round(last_pos, 2)
        save_gt(gt_data)
        print(f"\n  Quit without saving events. Position saved for --resume.")


if __name__ == "__main__":
    main()
