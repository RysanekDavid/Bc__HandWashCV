"""
Compute hand-wash statistics from detector output for thesis and company reporting.

Reads a detection CSV (from compare_detectors or yolo_cls_detector) and optional
ground-truth JSON, then produces:
  - Wash count per hour
  - Duration distribution (histogram buckets)
  - Per-station breakdown
  - Gap analysis (longest period without washing)
  - WHO compliance (washes >= 20s)
  - Hourly heatmap data

Usage:
    python src/compute_statistics.py outputs/detections/yolov8n_20260127_234623_tp00003.csv
    python src/compute_statistics.py outputs/detections/yolov8n_20260127_234623_tp00003.csv --gt outputs/ground_truth/full_video_gt.json
    python src/compute_statistics.py outputs/detections/yolov8n_20260127_234623_tp00003.csv --video-start "2026-01-27 23:46:23"
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EVAL_DIR


# ── Duration buckets ─────────────────────────────────────────

DURATION_BUCKETS = [
    (0, 5, "<5s (rinse)"),
    (5, 10, "5-10s (short)"),
    (10, 20, "10-20s (moderate)"),
    (20, 30, "20-30s (WHO compliant)"),
    (30, 60, "30-60s (thorough)"),
    (60, float("inf"), ">60s (extended)"),
]


def bucket_durations(durations: list[float]) -> list[dict]:
    """Assign durations to predefined buckets."""
    results = []
    for lo, hi, label in DURATION_BUCKETS:
        count = sum(1 for d in durations if lo <= d < hi)
        results.append({"range": label, "count": count, "pct": count / max(1, len(durations)) * 100})
    return results


# ── Core statistics ──────────────────────────────────────────

def compute_stats(events: list[dict], video_duration_sec: float | None = None,
                  video_start: datetime | None = None) -> dict:
    """Compute all statistics from a list of detection events."""
    if not events:
        return {"error": "No events to analyze"}

    durations = [e["end_sec"] - e["start_sec"] for e in events]
    starts = [e["start_sec"] for e in events]
    stations = [e.get("station", 0) for e in events]

    total_events = len(events)
    total_duration = video_duration_sec or (max(e["end_sec"] for e in events) + 60)

    # Basic stats
    stats = {
        "total_events": total_events,
        "video_duration_min": round(total_duration / 60, 1),
        "events_per_hour": round(total_events / (total_duration / 3600), 1),
    }

    # Duration stats
    stats["duration"] = {
        "mean_sec": round(sum(durations) / len(durations), 1),
        "median_sec": round(sorted(durations)[len(durations) // 2], 1),
        "min_sec": round(min(durations), 1),
        "max_sec": round(max(durations), 1),
        "std_sec": round(_std(durations), 1),
        "distribution": bucket_durations(durations),
    }

    # WHO compliance (>= 20 seconds)
    who_compliant = sum(1 for d in durations if d >= 20)
    stats["who_compliance"] = {
        "compliant_count": who_compliant,
        "total_count": total_events,
        "compliance_pct": round(who_compliant / total_events * 100, 1),
        "threshold_sec": 20,
    }

    # Per-station breakdown
    unique_stations = sorted(set(stations))
    station_stats = []
    for sid in unique_stations:
        s_events = [e for e, s in zip(events, stations) if s == sid]
        s_durations = [e["end_sec"] - e["start_sec"] for e in s_events]
        occupied_sec = sum(s_durations)
        s_who = sum(1 for d in s_durations if d >= 20)
        station_stats.append({
            "station": sid,
            "event_count": len(s_events),
            "mean_duration_sec": round(sum(s_durations) / len(s_durations), 1),
            "event_share_pct": round(len(s_events) / total_events * 100, 1),
            "occupied_sec": round(occupied_sec, 1),
            "occupied_pct": round(occupied_sec / total_duration * 100, 2),
            "who_compliant_pct": round(s_who / max(1, len(s_events)) * 100, 1),
        })
    stats["per_station"] = station_stats

    # Simultaneous usage (both stations occupied at the same time)
    if len(unique_stations) >= 2:
        sorted_ev = sorted(events, key=lambda e: e["start_sec"])
        overlap_sec = 0.0
        overlap_count = 0
        for i, a in enumerate(sorted_ev):
            for b in sorted_ev[i + 1:]:
                if b["start_sec"] >= a["end_sec"]:
                    break
                if a.get("station", 0) != b.get("station", 0):
                    ov_start = max(a["start_sec"], b["start_sec"])
                    ov_end = min(a["end_sec"], b["end_sec"])
                    if ov_end > ov_start:
                        overlap_sec += ov_end - ov_start
                        overlap_count += 1
        stats["simultaneous_usage"] = {
            "overlap_count": overlap_count,
            "overlap_total_sec": round(overlap_sec, 1),
            "overlap_pct_of_video": round(overlap_sec / total_duration * 100, 2),
        }

    # Gap analysis
    sorted_events = sorted(events, key=lambda e: e["start_sec"])
    gaps = []
    for i in range(1, len(sorted_events)):
        gap = sorted_events[i]["start_sec"] - sorted_events[i - 1]["end_sec"]
        gaps.append({
            "after_event": i,
            "gap_sec": round(gap, 1),
            "at_sec": round(sorted_events[i - 1]["end_sec"], 1),
        })

    if gaps:
        gaps_sorted = sorted(gaps, key=lambda g: g["gap_sec"], reverse=True)
        stats["gaps"] = {
            "longest_gap_sec": gaps_sorted[0]["gap_sec"],
            "longest_gap_at_sec": gaps_sorted[0]["at_sec"],
            "mean_gap_sec": round(sum(g["gap_sec"] for g in gaps) / len(gaps), 1),
            "top_5_gaps": gaps_sorted[:5],
        }

    # Hourly heatmap (bucket events into 5-minute bins)
    bin_size = 300  # 5 minutes
    n_bins = int(total_duration / bin_size) + 1
    bins = [0] * n_bins
    for s in starts:
        idx = min(int(s / bin_size), n_bins - 1)
        bins[idx] += 1

    heatmap = []
    for i, count in enumerate(bins):
        t = i * bin_size
        label = f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}"
        if video_start:
            abs_time = video_start + timedelta(seconds=t)
            label = abs_time.strftime("%H:%M")
        heatmap.append({"time_label": label, "offset_sec": t, "events": count})
    stats["heatmap_5min"] = heatmap

    # Real-time labels (if video start time known)
    if video_start:
        stats["real_time_events"] = []
        for e in sorted_events:
            abs_start = video_start + timedelta(seconds=e["start_sec"])
            abs_end = video_start + timedelta(seconds=e["end_sec"])
            stats["real_time_events"].append({
                "start": abs_start.strftime("%H:%M:%S"),
                "end": abs_end.strftime("%H:%M:%S"),
                "duration_sec": round(e["end_sec"] - e["start_sec"], 1),
                "station": e.get("station", 0),
            })

    return stats


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ── Formatted output ────────────────────────────────────────

def print_report(stats: dict) -> None:
    """Print a formatted statistics report."""
    print(f"\n{'=' * 65}")
    print(f"  HAND-WASH STATISTICS REPORT")
    print(f"{'=' * 65}")

    print(f"\n  Overview")
    print(f"  {'Total wash events:':<30s} {stats['total_events']}")
    print(f"  {'Video duration:':<30s} {stats['video_duration_min']} min")
    print(f"  {'Events per hour:':<30s} {stats['events_per_hour']}")

    d = stats["duration"]
    print(f"\n  Wash Duration")
    print(f"  {'Mean:':<30s} {d['mean_sec']}s")
    print(f"  {'Median:':<30s} {d['median_sec']}s")
    print(f"  {'Range:':<30s} {d['min_sec']}s - {d['max_sec']}s")
    print(f"  {'Std deviation:':<30s} {d['std_sec']}s")

    print(f"\n  Duration Distribution")
    for b in d["distribution"]:
        bar = "#" * int(b["pct"] / 2)
        print(f"    {b['range']:<26s} {b['count']:>3d} ({b['pct']:5.1f}%) {bar}")

    w = stats["who_compliance"]
    emoji = "PASS" if w["compliance_pct"] >= 80 else "WARN"
    print(f"\n  WHO Compliance (>= {w['threshold_sec']}s)")
    print(f"  {'Compliant washes:':<30s} {w['compliant_count']}/{w['total_count']} ({w['compliance_pct']}%) [{emoji}]")

    print(f"\n  Per-Station Breakdown")
    for s in stats["per_station"]:
        print(f"    Station {s['station']}: {s['event_count']} events "
              f"({s['event_share_pct']}% of total), "
              f"mean {s['mean_duration_sec']}s")
        print(f"      Occupied: {s['occupied_sec']}s ({s['occupied_pct']}% of video), "
              f"WHO compliant: {s['who_compliant_pct']}%")

    if "simultaneous_usage" in stats:
        su = stats["simultaneous_usage"]
        print(f"\n  Simultaneous Usage (both stations at once)")
        print(f"  {'Overlapping events:':<30s} {su['overlap_count']}")
        print(f"  {'Total overlap time:':<30s} {su['overlap_total_sec']}s")
        print(f"  {'% of video duration:':<30s} {su['overlap_pct_of_video']}%")

    if "gaps" in stats:
        g = stats["gaps"]
        print(f"\n  Gap Analysis")
        print(f"  {'Longest gap:':<30s} {g['longest_gap_sec']}s ({g['longest_gap_sec']/60:.1f} min) "
              f"at {_fmt_sec(g['longest_gap_at_sec'])}")
        print(f"  {'Mean gap between washes:':<30s} {g['mean_gap_sec']}s ({g['mean_gap_sec']/60:.1f} min)")

    if "person_compliance" in stats:
        c = stats["person_compliance"]
        print(f"\n  Person Compliance (exit tracking)")
        print(f"  {'Total person exits:':<30s} {c['total_exits']}")
        print(f"  {'Washed before exit:':<30s} {c['compliant']}")
        print(f"  {'Skipped washing:':<30s} {c['non_compliant']}")
        print(f"  {'Compliance rate:':<30s} {c['compliance_rate_pct']}%")
        print(f"  {'Exits per hour:':<30s} {c['exits_per_hour']}")

    print(f"\n{'=' * 65}")


def _fmt_sec(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute hand-wash statistics.")
    parser.add_argument("csv", help="Detection CSV file (from compare_detectors or yolo_cls_detector).")
    parser.add_argument("--gt", default=None, help="Ground-truth JSON for comparison context.")
    parser.add_argument("--video-start", default=None,
                        help="Video start time for absolute timestamps (e.g. '2026-01-27 23:46:23').")
    parser.add_argument("--video-duration", type=float, default=None,
                        help="Override video duration in seconds.")
    parser.add_argument("--compliance", default=None,
                        help="Compliance report JSON (from person_tracker.py) to include in output.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(str(csv_path))
    events = df.to_dict("records")
    print(f"Loaded {len(events)} events from {csv_path.name}")

    video_start = None
    if args.video_start:
        video_start = datetime.strptime(args.video_start, "%Y-%m-%d %H:%M:%S")

    video_duration = args.video_duration

    # Try to get video duration from GT if available
    if args.gt:
        gt_data = json.load(open(args.gt, encoding="utf-8"))
        if not video_duration and "last_position_sec" in gt_data:
            video_duration = gt_data["last_position_sec"]
        gt_count = len(gt_data.get("events", []))
        print(f"GT reference: {gt_count} events")

    stats = compute_stats(events, video_duration_sec=video_duration, video_start=video_start)

    # Merge compliance report if provided
    if args.compliance:
        comp_path = Path(args.compliance)
        if comp_path.exists():
            comp = json.load(open(comp_path, encoding="utf-8"))
            stats["person_compliance"] = comp["summary"]
            print(f"Compliance report loaded: {comp['summary']['compliance_rate_pct']}% "
                  f"({comp['summary']['compliant']}/{comp['summary']['total_exits']} exits)")

    print_report(stats)

    # Save JSON
    out_path = EVAL_DIR / f"statistics_{csv_path.stem}.json"
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStatistics saved: {out_path}")


if __name__ == "__main__":
    main()
