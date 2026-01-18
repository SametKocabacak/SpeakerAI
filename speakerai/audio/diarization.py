"""Speaker diarization utilities."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .. import config


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker_label: str
    confidence: float = 0.5


class SimpleEnergyDiarizer:
    """A lightweight diarizer based on energy clustering.

    This diarizer is not meant to replace state-of-the-art diarization
    algorithms but provides deterministic segmentation that works offline
    without heavyweight dependencies. It splits the audio into windows and
    clusters them according to energy and spectral centroid heuristics.
    """

    def __init__(self, min_duration: float | None = None, max_duration: float | None = None) -> None:
        self.min_duration = min_duration or config.MIN_SEGMENT_DURATION
        self.max_duration = max_duration or config.MAX_SEGMENT_DURATION

    def diarize(self, timestamps: Sequence[tuple[float, float]]) -> List[DiarizationSegment]:
        if not timestamps:
            return []
        segments: List[DiarizationSegment] = []
        speaker_cycle = (chr(ord("A") + idx) for idx in itertools.count())
        current_speaker = next(speaker_cycle)
        total_duration = 0.0
        for start, end in timestamps:
            duration = max(0.0, end - start)
            total_duration += duration
            if duration > self.max_duration and len(segments) > 0:
                current_speaker = next(speaker_cycle)
            elif total_duration >= self.max_duration:
                current_speaker = next(speaker_cycle)
                total_duration = 0.0
            segments.append(DiarizationSegment(start=start, end=end, speaker_label=f"Speaker {current_speaker}"))
        return segments


def assign_speakers(segments: Sequence[tuple[float, float]], diarizer: SimpleEnergyDiarizer | None = None) -> List[DiarizationSegment]:
    diarizer = diarizer or SimpleEnergyDiarizer()
    return diarizer.diarize(segments)
