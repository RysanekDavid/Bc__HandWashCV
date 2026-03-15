"""
Interactive video annotation tool for hand-wash events.

Controls:
    SPACE       Pause / Resume
    S           Mark wash START at current time
    E           Mark wash END   at current time
    U           Undo last mark
    D           Delete last completed event
    Q           Save annotations & go to next clip
    ESC         Quit without saving current clip
    RIGHT (→)   Skip forward  2 seconds
    LEFT  (←)   Skip backward 2 seconds
    +/-         Speed up / slow down playback

Output:  outputs/annotations.json  (merged across sessions)

Usage:
    python src/annotate.py                              # all clips
    python src/annotate.py --clip clip_0064.mp4         # single clip
    python src/annotate.py --limit 20                   # first 20 clips
    python src/annotate.py --skip-annotated             # skip already labeled clips
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import UNLABELED_DIR, LABELED_DIR, OUTPUTS_DIR, DEFAULT_ROI_PATH
from roi_select import load_roi


ANNOTATIONS_PATH = OUTPUTS_DIR / "annotations.json"

WINDOW_NAME = "Annotate (S=start, E=end, U=undo, D=del, Q=save+next, ESC=quit)"

SPEED_OPTIONS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
DEFAULT_SPEED_IDX = 2  # 1.0x


# ── Persistence ───────────────────────────────────────────────

def load_annotations() -> dict:
    """Load existing annotations or return empty dict."""
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(data: dict) -> None:
    """Persist annotations dict to JSON."""
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Drawing helpers ───────────────────────────────────────────

def _draw_hud(
    frame: np.ndarray,
    time_sec: float,
    total_sec: float,
    paused: bool,
    speed: float,
    events: list[dict],
    pending_start: Optional[float],
    roi: Optional[dict],
    clip_info: str = "",
) -> None:
    """Draw time, status, events list, and ROI onto frame."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Clip progress (top-right)
    if clip_info:
        cv2.putText(frame, clip_info, (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Time + status
    status = "PAUSED" if paused else f"PLAYING {speed:.1f}x"
    time_str = f"{_fmt(time_sec)} / {_fmt(total_sec)}"
    cv2.putText(frame, time_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Progress bar
    bar_y = 80
    bar_w = w - 30
    progress = time_sec / total_sec if total_sec > 0 else 0
    cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + 6), (80, 80, 80), -1)
    cv2.rectangle(frame, (15, bar_y), (15 + int(bar_w * progress), bar_y + 6), (0, 255, 255), -1)

    # Draw event markers on progress bar
    for ev in events:
        sx = 15 + int((ev["start_sec"] / total_sec) * bar_w)
        ex = 15 + int((ev["end_sec"] / total_sec) * bar_w)
        cv2.rectangle(frame, (sx, bar_y - 2), (ex, bar_y + 8), (0, 255, 0), -1)

    # Pending start marker
    if pending_start is not None:
        px = 15 + int((pending_start / total_sec) * bar_w)
        cv2.line(frame, (px, bar_y - 4), (px, bar_y + 10), (0, 0, 255), 2)
        cv2.putText(
            frame, f"START @ {_fmt(pending_start)}", (w - 300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    # Events list (bottom-left)
    y_off = h - 20
    for i, ev in enumerate(reversed(events)):
        txt = f"Event {len(events) - i}: {_fmt(ev['start_sec'])} -> {_fmt(ev['end_sec'])}"
        cv2.putText(frame, txt, (15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_off -= 22
        if y_off < h - 200:
            break

    # ROI rectangle
    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 0), 1)

    # Controls reminder (bottom-right)
    cv2.putText(
        frame, "S=start E=end U=undo D=del Q=save ESC=quit",
        (w - 520, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
    )


def _fmt(sec: float) -> str:
    """Format seconds as MM:SS.d"""
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


# ── Core annotation loop ─────────────────────────────────────

def annotate_clip(
    video_path: Path,
    roi: Optional[dict] = None,
    existing_events: Optional[list[dict]] = None,
    clip_info: str = "",
) -> Optional[list[dict]]:
    """
    Play video and let user mark wash events interactively.
    Returns list of events, or None if user pressed ESC (skip/quit).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    events: list[dict] = list(existing_events) if existing_events else []
    pending_start: Optional[float] = None

    paused = False
    speed_idx = DEFAULT_SPEED_IDX

    print(f"\n--- Annotating: {video_path.name} ({_fmt(total_sec)}) ---")
    if events:
        print(f"    Loaded {len(events)} existing event(s)")

    _frame_t0 = time.perf_counter()
    while True:
        _frame_t0 = time.perf_counter()
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # End of video — auto-pause at last frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                paused = True
        else:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_sec = current_frame / fps

        _draw_hud(frame, current_sec, total_sec, paused,
                   SPEED_OPTIONS[speed_idx], events, pending_start, roi,
                   clip_info)

        cv2.imshow(WINDOW_NAME, frame)

        target_delay = (1000 / fps) / SPEED_OPTIONS[speed_idx]
        elapsed_ms = (time.perf_counter() - _frame_t0) * 1000
        wait_ms = max(1, int(target_delay - elapsed_ms))
        key = cv2.waitKey(wait_ms if not paused else 0) & 0xFF

        # ── Key handling ──────────────────────────────

        if key == 27:  # ESC — quit without saving
            cap.release()
            cv2.destroyAllWindows()
            return None

        elif key == ord("q") or key == ord("Q"):  # Save & next
            cap.release()
            cv2.destroyAllWindows()
            if pending_start is not None:
                print(f"    WARNING: Discarded pending start @ {_fmt(pending_start)}")
            return events

        elif key == ord(" "):  # Pause / Resume
            paused = not paused

        elif key == ord("s") or key == ord("S"):  # Mark START
            pending_start = current_sec
            print(f"    START marked @ {_fmt(current_sec)}")

        elif key == ord("e") or key == ord("E"):  # Mark END
            if pending_start is None:
                print("    WARNING: No start marked yet, press S first")
            else:
                end_sec = current_sec
                if end_sec <= pending_start:
                    print("    WARNING: End must be after start, ignoring")
                else:
                    ev = {
                        "start_sec": round(pending_start, 2),
                        "end_sec": round(end_sec, 2),
                    }
                    events.append(ev)
                    print(f"    EVENT added: {_fmt(ev['start_sec'])} -> {_fmt(ev['end_sec'])}")
                    pending_start = None

        elif key == ord("u") or key == ord("U"):  # Undo pending start
            if pending_start is not None:
                print(f"    Undo: cleared pending start @ {_fmt(pending_start)}")
                pending_start = None
            else:
                print("    Nothing to undo")

        elif key == ord("d") or key == ord("D"):  # Delete last event
            if events:
                removed = events.pop()
                print(f"    Deleted last event: {_fmt(removed['start_sec'])} -> {_fmt(removed['end_sec'])}")
            else:
                print("    No events to delete")

        elif key == ord("+") or key == ord("="):  # Speed up
            speed_idx = min(speed_idx + 1, len(SPEED_OPTIONS) - 1)

        elif key == ord("-"):  # Slow down
            speed_idx = max(speed_idx - 1, 0)

        elif key == 83 or key == ord("l"):  # RIGHT arrow → skip forward 2s
            new_frame = min(current_frame + int(2 * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

        elif key == 81 or key == ord("h"):  # LEFT arrow → skip backward 2s
            new_frame = max(current_frame - int(2 * fps), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    cap.release()
    cv2.destroyAllWindows()
    return events


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive hand-wash annotation tool.")
    parser.add_argument("--clip", default=None, help="Annotate a single clip (filename).")
    parser.add_argument("--limit", type=int, default=None, help="Annotate first N clips.")
    parser.add_argument("--skip-annotated", action="store_true",
                        help="Skip clips that already have annotations.")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH), help="ROI JSON path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roi = None
    roi_path = Path(args.roi)
    if roi_path.exists():
        roi = load_roi(roi_path)

    annotations = load_annotations()

    # Resolve clip list
    if args.clip:
        clip_path = UNLABELED_DIR / args.clip
        if not clip_path.exists():
            # Check labeled just in case
            clip_path = LABELED_DIR / args.clip
            if not clip_path.exists():
                print(f"ERROR: Clip not found: {args.clip}")
                sys.exit(1)
        clips = [clip_path]
    else:
        # We only want to label things in the UNLABELED folder
        clips = sorted(UNLABELED_DIR.glob("clip_*.mp4"))
        if args.limit:
            clips = clips[: args.limit]

    if not clips:
        # Check if we are done or if folder is just empty
        if not args.clip and not list(UNLABELED_DIR.iterdir()):
            print(f"DONE! No more clips to annotate in {UNLABELED_DIR}")
            sys.exit(0)
        print(f"ERROR: No clips found in {UNLABELED_DIR}")
        sys.exit(1)

    print(f"Clips to annotate: {len(clips)}")
    if args.skip_annotated:
        print(f"Already annotated : {sum(1 for c in clips if c.name in annotations)}")

    for clip in clips:
        name = clip.name

        if args.skip_annotated and name in annotations:
            print(f"Skipping (already annotated): {name}")
            continue

        existing = annotations.get(name, {}).get("events", [])

        idx = clips.index(clip) + 1
        info = f"[{idx}/{len(clips)}] {name}"
        result = annotate_clip(clip, roi=roi, existing_events=existing, clip_info=info)

        if result is None:
            print("ESC pressed — quitting. Progress so far is saved.")
            break

        annotations[name] = {"events": result}
        save_annotations(annotations)
        print(f"    Saved {len(result)} event(s) for {name}")

        # Automate move to labeled directory
        LABELED_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = LABELED_DIR / name
        try:
            import shutil
            shutil.move(str(clip), str(dest_path))
            print(f"    MOVED: {name} -> data_clips/labeled/")
        except Exception as e:
            print(f"    ERROR moving file: {e}")

    print(f"\nAnnotations file: {ANNOTATIONS_PATH}")
    print(f"Total clips annotated: {sum(1 for v in annotations.values() if 'events' in v)}")


if __name__ == "__main__":
    main()
