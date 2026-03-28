import subprocess
import argparse
from pathlib import Path

# --- Configuration ---
from config import UNLABELED_DIR, PROJECT_ROOT

RAW_VIDEOS_DIR = PROJECT_ROOT / "data_clips" / "2026-02-06"
OUTPUT_DIR = UNLABELED_DIR
CLIP_DURATION = 20  # seconds

# Path to local ffmpeg and ffprobe
FFMPEG_PATH = PROJECT_ROOT / "ffmpeg-8.0.1-essentials_build" / "bin" / "ffmpeg.exe"
FFPROBE_PATH = PROJECT_ROOT / "ffmpeg-8.0.1-essentials_build" / "bin" / "ffprobe.exe"

# List of (video_name, list_of_start_times_in_seconds)
# Use None instead of a list to cut the ENTIRE video into segments
DEFAULT_CLIPS_TO_GENERATE = [
    ("20260127_193759_tp00002.mp4", None),
]

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        str(FFPROBE_PATH), "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def cut_clip(video_name, start_sec, clip_id, dry_run=False):
    input_path = RAW_VIDEOS_DIR / video_name
    if not input_path.exists():
        print(f"Error: Video {video_name} not found in {RAW_VIDEOS_DIR}")
        return False
        
    output_name = f"clip_{clip_id:04d}.mp4"
    output_path = OUTPUT_DIR / output_name
    
    start_time_str = format_time(start_sec)
    
    # Use the "safe" command for correct timestamps
    cmd = [
        str(FFMPEG_PATH), "-y",
        "-ss", start_time_str,
        "-i", str(input_path),
        "-t", str(CLIP_DURATION),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]
    
    print(f"Generating {output_name} from {video_name} at {start_time_str}...")
    
    if dry_run:
        print("  [DRY-RUN] " + " ".join(cmd))
        return True
    else:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Error cutting clip: {e.stderr.decode()}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Batch cut clips from raw videos.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--start-id", type=int, default=100, help="Starting ID for clip filenames.")
    parser.add_argument("--interval", type=int, default=None, help="Interval in seconds (defaults to CLIP_DURATION if auto-cutting).")
    
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    current_id = args.start_id
    success_count = 0
    interval = args.interval or CLIP_DURATION
    
    for video_name, start_times in DEFAULT_CLIPS_TO_GENERATE:
        input_path = RAW_VIDEOS_DIR / video_name
        
        # If no specific times, generate segments for the whole video
        if start_times is None:
            duration = get_video_duration(input_path)
            if duration == 0:
                continue
            # Calculate how many full segments fit
            num_segments = int(duration // interval)
            print(f"Video {video_name} duration: {duration:.2f}s. Cutting into {num_segments} segments...")
            start_times = [i * interval for i in range(num_segments)]
            
        for t in start_times:
            if cut_clip(video_name, t, current_id, dry_run=args.dry_run):
                success_count += 1
                current_id += 1
    
    print(f"\nDone! Successfully {'prepared' if args.dry_run else 'generated'} {success_count} clips.")
    if not args.dry_run:
        print(f"Check the output in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
