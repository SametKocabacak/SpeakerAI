"""Audio transcription utilities."""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import whisper  # type: ignore
except Exception:  # pragma: no cover - fallback
    whisper = None

from .. import config
from ..models import SpeakerTurn
from .diarization import DiarizationSegment, assign_speakers

LOGGER = logging.getLogger(__name__)


def ensure_ffmpeg() -> Path:
    """Ensure an FFmpeg binary is available and return its path."""

    existing = shutil.which("ffmpeg")
    if existing:
        return Path(existing)

    try:  # pragma: no cover - optional dependency
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
    except Exception as exc:  # pragma: no cover - graceful error handling
        raise RuntimeError(
            "FFmpeg is required but was not found. Install the system package or add "
            "`imageio-ffmpeg` to your environment."
        ) from exc

    ffmpeg_path = Path(get_ffmpeg_exe())
    os.environ["PATH"] = f"{ffmpeg_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return ffmpeg_path


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0


class TranscriptionService:
    """Convert audio files to timestamped transcripts."""

    def __init__(self, model_name: str = "small", language: Optional[str] = None) -> None:
        self.model_name = model_name
        self.language = language
        self._model = None

    def _load_model(self):  # pragma: no cover - whisper is heavy to import during tests
        if whisper is None:
            raise RuntimeError(
                "The 'whisper' package is required for transcription. Install it via `pip install openai-whisper`."
            )
        ensure_ffmpeg()
        if self._model is None:
            LOGGER.info("Loading Whisper model '%s'", self.model_name)
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio_path: str | Path) -> List[SpeakerTurn]:
        audio_path = Path(audio_path)
        segments = self._transcribe_segments(audio_path)
        diarized_segments = assign_speakers([(seg.start, seg.end) for seg in segments])
        turns: List[SpeakerTurn] = []
        for seg, diarized in zip(segments, diarized_segments):
            turns.append(
                SpeakerTurn(
                    speaker_label=diarized.speaker_label,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    confidence=seg.confidence,
                )
            )
        return turns

    def _transcribe_segments(self, audio_path: Path) -> List[TranscriptSegment]:
        try:
            model = self._load_model()
        except RuntimeError as exc:
            LOGGER.warning("Falling back to naive transcription: %s", exc)
            return self._naive_transcription(audio_path)
        result = model.transcribe(str(audio_path), language=self.language, word_timestamps=True)
        segments: List[TranscriptSegment] = []
        for segment in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=segment.get("text", ""),
                    confidence=float(segment.get("avg_logprob", 0.0)),
                )
            )
        return segments

    def _naive_transcription(self, audio_path: Path) -> List[TranscriptSegment]:
        LOGGER.warning("Performing naive segmentation for %s", audio_path)
        # Without a speech recognition backend we fall back to silence; this
        # allows development/testing without heavy dependencies.
        duration = self._estimate_duration(audio_path)
        segments: List[TranscriptSegment] = []
        start = 0.0
        idx = 0
        while start < duration:
            end = min(duration, start + config.MAX_SEGMENT_DURATION)
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=f"[UNTRANSCRIBED SEGMENT {idx}]",
                    confidence=0.0,
                )
            )
            idx += 1
            start = end
        return segments

    def _estimate_duration(self, audio_path: Path) -> float:
        try:
            import librosa  # type: ignore  # pragma: no cover

            y, sr = librosa.load(audio_path, sr=config.TARGET_SAMPLE_RATE)
            return float(len(y) / sr)
        except Exception:
            LOGGER.error("librosa is required for duration estimation. Assuming 60 seconds.")
            return 60.0
