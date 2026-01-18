"""Audio feature extraction for speaker characterization."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional heavy dependency for tests
    import numpy as np
except Exception:  # pragma: no cover
    import math

    class _NPModule:
        @staticmethod
        def array(values):
            return [float(v) for v in values]

        @staticmethod
        def mean(values):
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        @staticmethod
        def std(values):
            if not values:
                return 0.0
            m = _NPModule.mean(values)
            return float(math.sqrt(sum((v - m) ** 2 for v in values) / len(values)))

        @staticmethod
        def median(values):
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mid = n // 2
            if n % 2 == 0:
                return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2)
            return float(sorted_vals[mid])

        @staticmethod
        def sqrt(value):
            return float(math.sqrt(value))

    np = _NPModule()  # type: ignore

from .. import config

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    mean_pitch: float
    pitch_std: float
    speaking_rate: float
    energy_mean: float
    energy_std: float
    spectral_centroid_mean: float
    spectral_centroid_std: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_pitch": self.mean_pitch,
            "pitch_std": self.pitch_std,
            "speaking_rate": self.speaking_rate,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "spectral_centroid_mean": self.spectral_centroid_mean,
            "spectral_centroid_std": self.spectral_centroid_std,
        }


class FeatureExtractor:
    def __init__(self, sample_rate: int = config.TARGET_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        import librosa  # type: ignore

        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr

    def compute_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa  # type: ignore

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch = pitches[magnitudes > np.median(magnitudes)]
        if pitch.size == 0:
            return np.array([0.0])
        return pitch

    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        frame_length = int(0.05 * self.sample_rate)
        hop_length = frame_length // 2
        energy = []
        for idx in range(0, len(audio), hop_length):
            frame = audio[idx : idx + frame_length]
            if frame.size == 0:
                continue
            energy.append(float(np.sqrt(np.mean(frame**2))))
        return np.array(energy) if energy else np.array([0.0])

    def compute_speaking_rate(self, transcript_texts: Iterable[str], duration: float) -> float:
        total_words = sum(len(text.split()) for text in transcript_texts)
        if duration <= 0:
            return 0.0
        return total_words / (duration / 60.0)

    def spectral_centroid(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa  # type: ignore

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return centroid.flatten()

    def extract_features(self, audio_path: Path, transcript_texts: Iterable[str], start: float, end: float) -> FeatureSet:
        y, sr = self.load_audio(audio_path)
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        segment_audio = y[start_idx:end_idx]
        if segment_audio.size == 0:
            segment_audio = y
        pitch = self.compute_pitch(segment_audio, sr)
        energy = self.compute_energy(segment_audio)
        centroid = self.spectral_centroid(segment_audio, sr)
        speaking_rate = self.compute_speaking_rate(transcript_texts, max(1.0, end - start))
        return FeatureSet(
            mean_pitch=float(np.mean(pitch)),
            pitch_std=float(np.std(pitch)),
            speaking_rate=float(speaking_rate),
            energy_mean=float(np.mean(energy)),
            energy_std=float(np.std(energy)),
            spectral_centroid_mean=float(np.mean(centroid)),
            spectral_centroid_std=float(np.std(centroid)),
        )
