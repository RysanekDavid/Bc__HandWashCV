"""
Centralized configuration for the hand-wash detection pipeline.
All default paths and detection parameters live here.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_CLIPS_DIR = PROJECT_ROOT / "data_clips"
UNLABELED_DIR = DATA_CLIPS_DIR / "unlabeled"
LABELED_DIR = DATA_CLIPS_DIR / "labeled"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_ROI_PATH = OUTPUTS_DIR / "roi.json"


# ── Detection parameters ──────────────────────────────────────

@dataclass
class DetectionParams:
    """Tuneable parameters for the baseline motion detector."""

    motion_thresh: int = 2500
    """Minimum foreground-pixel count to consider as 'motion'."""

    wash_sec_on: float = 5.0
    """Seconds of continuous motion required to START a wash event."""

    wash_sec_off: float = 2.0
    """Seconds of continuous stillness required to END a wash event."""

    bg_history: int = 500
    """MOG2 background-subtractor history length (frames)."""

    bg_var_threshold: float = 32.0
    """MOG2 variance threshold."""

    median_blur_k: int = 5
    """Kernel size for median-blur denoising of the foreground mask."""
