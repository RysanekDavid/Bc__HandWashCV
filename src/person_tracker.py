"""
Person tracking + hand-wash compliance monitoring.

Uses YOLO person detection with built-in ByteTrack to:
  1. Detect and track people near the wash stations
  2. Detect when a tracked person enters/exits the monitoring zone
  3. Correlate exits with wash events (from yolo_cls_detector or soap_trigger)
  4. Compute compliance rate and per-person statistics

The exit zone is defined in roi.json under "exit_zone" key.
Use roi_select.py --exit-zone to define it interactively.

Usage:
    python src/person_tracker.py <video> --wash-csv <detections.csv> [--no-preview]

Output:
    outputs/compliance_report.json — per-person wash correlation + aggregate stats
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, EVAL_DIR, MODELS_DIR, DEFAULT_ROI_PATH, PROJECT_ROOT


# ── Person track record ──────────────────────────────────────

@dataclass
class PersonTrack:
    """State for one tracked person."""
    track_id: int
    first_seen_sec: float = 0.0
    last_seen_sec: float = 0.0
    first_pos: tuple[int, int] = (0, 0)
    last_pos: tuple[int, int] = (0, 0)
    positions: list[tuple[float, int, int]] = field(default_factory=list)
    in_zone: bool = False
    exited: bool = False
    exit_sec: float = 0.0
    washed: bool = False
    wash_event: dict | None = None


# ── Zone checks ──────────────────────────────────────────────

def point_in_rect(px: int, py: int, zone: dict) -> bool:
    return (zone["x"] <= px <= zone["x"] + zone["w"]
            and zone["y"] <= py <= zone["y"] + zone["h"])


def bbox_center(box) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# ── Wash event correlation ───────────────────────────────────

def find_wash_for_exit(exit_sec: float, wash_events: list[dict],
                       lookback_sec: float = 60.0) -> dict | None:
    """Find the most recent wash event before this person's exit."""
    best = None
    for ev in wash_events:
        # Wash must have ended before or near the exit time
        if ev["end_sec"] <= exit_sec + 5 and ev["end_sec"] >= exit_sec - lookback_sec:
            if best is None or ev["end_sec"] > best["end_sec"]:
                best = ev
    return best


# ── Main tracking function ───────────────────────────────────

def track_persons(
    video_path: str,
    roi: dict,
    exit_zone: dict,
    wash_events: list[dict],
    show_preview: bool = True,
    person_model: str = str(MODELS_DIR / "yolov8n.pt"),
    lookback_sec: float = 60.0,
) -> dict:
    """
    Track persons and compute compliance.

    Returns dict with:
      - persons: list of per-person records
      - compliance: aggregate compliance stats
    """
    from ultralytics import YOLO

    model = YOLO(person_model)
    print(f"Person model: {person_model}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fps:.0f} fps, {total_frames} frames, {total_frames/fps/60:.1f} min")

    tracks: dict[int, PersonTrack] = {}
    completed_exits: list[PersonTrack] = []

    # Process every 3rd frame for speed (person tracking doesn't need every frame)
    skip = 3
    frame_idx = 0
    report_interval = int(fps * 60)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_sec = frame_idx / fps

        if frame_idx % skip == 0:
            # Run YOLO tracking (class 0 = person, built-in ByteTrack)
            results = model.track(
                frame, persist=True, classes=[0],
                conf=0.3, iou=0.5,
                verbose=False, tracker="bytetrack.yaml",
            )

            # Process tracked persons
            active_ids = set()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, ids):
                    cx, cy = bbox_center(box)
                    active_ids.add(tid)

                    if tid not in tracks:
                        tracks[tid] = PersonTrack(
                            track_id=tid,
                            first_seen_sec=current_sec,
                            first_pos=(cx, cy),
                        )

                    t = tracks[tid]
                    t.last_seen_sec = current_sec
                    t.last_pos = (cx, cy)
                    t.positions.append((current_sec, cx, cy))

                    # Check zone transitions
                    was_in = t.in_zone
                    is_in = point_in_rect(cx, cy, roi)
                    t.in_zone = is_in

                    # Exit detection: was in ROI zone, now in exit zone
                    if not t.exited and point_in_rect(cx, cy, exit_zone):
                        t.exited = True
                        t.exit_sec = current_sec

                        # Correlate with wash events
                        wash = find_wash_for_exit(current_sec, wash_events, lookback_sec)
                        t.washed = wash is not None
                        t.wash_event = wash

                        completed_exits.append(t)
                        status = "COMPLIANT" if t.washed else "NON-COMPLIANT"
                        wash_info = ""
                        if wash:
                            dur = wash["end_sec"] - wash["start_sec"]
                            wash_info = f" (wash at {wash['start_sec']:.0f}s, {dur:.0f}s)"
                        print(f"  EXIT #{len(completed_exits)}: Person {tid} at "
                              f"{current_sec:.1f}s — {status}{wash_info}")

            # Draw preview
            if show_preview:
                _draw_tracking(frame, results, roi, exit_zone, tracks,
                               completed_exits, current_sec)
                cv2.imshow("Person Tracker + Compliance", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        frame_idx += 1
        if frame_idx % report_interval == 0:
            print(f"  {current_sec/60:.1f} min | tracks: {len(tracks)}, "
                  f"exits: {len(completed_exits)}")

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    return _build_report(completed_exits, wash_events, total_frames / fps)


# ── Report builder ───────────────────────────────────────────

def _build_report(exits: list[PersonTrack], wash_events: list[dict],
                  video_duration: float) -> dict:
    """Build compliance report from completed exits."""
    compliant = [e for e in exits if e.washed]
    non_compliant = [e for e in exits if not e.washed]

    rate = len(compliant) / max(1, len(exits)) * 100

    persons = []
    for e in exits:
        record = {
            "track_id": e.track_id,
            "exit_sec": round(e.exit_sec, 1),
            "time_in_zone_sec": round(e.last_seen_sec - e.first_seen_sec, 1),
            "washed": e.washed,
        }
        if e.wash_event:
            record["wash_start_sec"] = e.wash_event["start_sec"]
            record["wash_end_sec"] = e.wash_event["end_sec"]
            record["wash_duration_sec"] = round(
                e.wash_event["end_sec"] - e.wash_event["start_sec"], 1)
            record["wash_to_exit_sec"] = round(
                e.exit_sec - e.wash_event["end_sec"], 1)
        persons.append(record)

    # Wash durations for compliant persons
    wash_durations = [p["wash_duration_sec"] for p in persons if p.get("wash_duration_sec")]

    report = {
        "summary": {
            "total_exits": len(exits),
            "compliant": len(compliant),
            "non_compliant": len(non_compliant),
            "compliance_rate_pct": round(rate, 1),
            "video_duration_min": round(video_duration / 60, 1),
            "exits_per_hour": round(len(exits) / max(0.01, video_duration / 3600), 1),
        },
        "wash_stats": {
            "mean_duration_sec": round(sum(wash_durations) / max(1, len(wash_durations)), 1),
            "who_compliant_pct": round(
                sum(1 for d in wash_durations if d >= 20) / max(1, len(wash_durations)) * 100, 1),
        },
        "persons": persons,
    }

    return report


# ── Drawing ──────────────────────────────────────────────────

def _draw_tracking(frame, results, roi, exit_zone, tracks, exits, current_sec):
    """Draw tracking visualization."""
    # ROI zone
    rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(frame, "WASH ZONE", (rx, ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Exit zone
    ex, ey, ew, eh = exit_zone["x"], exit_zone["y"], exit_zone["w"], exit_zone["h"]
    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 165, 255), 2)
    cv2.putText(frame, "EXIT ZONE", (ex, ey - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Draw tracked persons
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            t = tracks.get(tid)
            if t and t.exited:
                color = (0, 200, 0) if t.washed else (0, 0, 255)
            else:
                color = (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Compliance counter
    n_comp = sum(1 for e in exits if e.washed)
    n_exits = len(exits)
    rate = n_comp / max(1, n_exits) * 100
    cv2.putText(frame, f"Exits: {n_exits}  Compliant: {n_comp} ({rate:.0f}%)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Person tracking + hand-wash compliance monitoring."
    )
    parser.add_argument("video", help="Path to input video.")
    parser.add_argument("--wash-csv", required=True,
                        help="Detection CSV from yolo_cls_detector or compare_detectors.")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH), help="ROI JSON path.")
    parser.add_argument("--person-model", default=str(MODELS_DIR / "yolov8n.pt"),
                        help="YOLO detection model for persons (default: models/yolov8n.pt).")
    parser.add_argument("--lookback", type=float, default=60.0,
                        help="Seconds to look back for wash event before exit (default: 60).")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Load wash events
    wash_csv = Path(args.wash_csv)
    if not wash_csv.exists():
        print(f"ERROR: Wash CSV not found: {wash_csv}")
        sys.exit(1)
    wash_df = pd.read_csv(str(wash_csv))
    wash_events = wash_df.to_dict("records")
    print(f"Loaded {len(wash_events)} wash events from {wash_csv.name}")

    # Load ROI + exit zone
    roi_data = json.load(open(args.roi, encoding="utf-8"))
    exit_zone = roi_data.get("exit_zone")
    if not exit_zone:
        print("WARNING: No 'exit_zone' in ROI JSON.")
        print("Define it by adding to outputs/roi.json:")
        print('  "exit_zone": {"x": ..., "y": ..., "w": ..., "h": ...}')
        print("\nOr run:  python src/roi_select.py --exit-zone")
        print("\nFalling back to interactive selection...")
        exit_zone = _select_exit_zone(str(video_path))
        if exit_zone:
            roi_data["exit_zone"] = exit_zone
            Path(args.roi).write_text(
                json.dumps(roi_data, indent=2), encoding="utf-8")
            print(f"Exit zone saved to {args.roi}")
        else:
            print("ERROR: Exit zone required for compliance tracking.")
            sys.exit(1)

    # Strip zone keys for plain ROI
    roi_data.pop("soap_zones", None)
    roi_data.pop("soap_zone", None)
    roi_data.pop("sink_zones", None)
    roi_data.pop("exit_zone", None)
    roi = roi_data

    print(f"\nStarting person tracking + compliance analysis...")
    report = track_persons(
        str(video_path), roi, exit_zone, wash_events,
        show_preview=not args.no_preview,
        person_model=args.person_model,
        lookback_sec=args.lookback,
    )

    # Print report
    s = report["summary"]
    print(f"\n{'=' * 60}")
    print(f"  COMPLIANCE REPORT")
    print(f"{'=' * 60}")
    print(f"  Total person exits:    {s['total_exits']}")
    print(f"  Compliant (washed):    {s['compliant']}")
    print(f"  Non-compliant:         {s['non_compliant']}")
    print(f"  Compliance rate:       {s['compliance_rate_pct']}%")
    print(f"  Exits per hour:        {s['exits_per_hour']}")
    if report["wash_stats"]["mean_duration_sec"] > 0:
        ws = report["wash_stats"]
        print(f"  Mean wash duration:    {ws['mean_duration_sec']}s")
        print(f"  WHO compliant (>=20s): {ws['who_compliant_pct']}%")
    print(f"{'=' * 60}")

    if report["persons"]:
        print(f"\n  Per-Person Detail:")
        for p in report["persons"]:
            status = "WASHED" if p["washed"] else "SKIPPED"
            detail = ""
            if p.get("wash_duration_sec"):
                detail = f" ({p['wash_duration_sec']}s wash, {p['wash_to_exit_sec']}s to exit)"
            print(f"    ID {p['track_id']:>4d} exit@{p['exit_sec']:>7.1f}s — {status}{detail}")

    # Save report
    out_path = EVAL_DIR / "compliance_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {out_path}")


def _select_exit_zone(video_path: str) -> dict | None:
    """Interactive exit zone selection from first frame."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    print("Draw exit zone rectangle and press ENTER (or C to cancel).")
    r = cv2.selectROI("Select EXIT ZONE", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    if r[2] == 0 or r[3] == 0:
        return None
    return {"x": int(r[0]), "y": int(r[1]), "w": int(r[2]), "h": int(r[3])}


if __name__ == "__main__":
    main()
