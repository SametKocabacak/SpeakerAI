#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

python - <<'PY'
from speakerai.audio.transcription import ensure_ffmpeg

try:
    path = ensure_ffmpeg()
    print(f"FFmpeg available at: {path}")
except Exception as exc:
    raise SystemExit(f"Failed to ensure FFmpeg availability: {exc}")
PY

echo "Environment is ready. Activate it with 'source .venv/bin/activate'."
