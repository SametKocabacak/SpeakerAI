"""Application configuration constants."""
from __future__ import annotations

from pathlib import Path

# Default directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
REPORT_DIR = OUTPUT_DIR / "reports"

# Database configuration
DEFAULT_DB_PATH = DATA_DIR / "speaker_profiles.sqlite3"

# Web configuration
WEB_HOST = "0.0.0.0"
WEB_PORT = 7883
UPLOAD_DIR = DATA_DIR / "uploads"

# Audio processing configuration
TARGET_SAMPLE_RATE = 16_000
MIN_SEGMENT_DURATION = 2.5
MAX_SEGMENT_DURATION = 15.0

# Matching configuration
FEATURE_SIMILARITY_THRESHOLD = 0.82

for directory in (DATA_DIR, OUTPUT_DIR, TRANSCRIPT_DIR, REPORT_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)
