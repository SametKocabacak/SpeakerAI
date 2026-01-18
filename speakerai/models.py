"""Data models for the SpeakerAI project."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional


@dataclass
class SpeakerTurn:
    """Represents a single speaking turn with diarization metadata."""

    speaker_label: str
    start: float
    end: float
    text: str
    speaker_name: Optional[str] = None
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "speaker_label": self.speaker_label,
            "speaker_name": self.speaker_name,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class SpeakerProfile:
    """Aggregated information about a speaker."""

    speaker_label: str
    features: Dict[str, float]
    stats: Dict[str, float]
    psychometrics: Dict[str, float]
    inferred_names: List[str] = field(default_factory=list)
    matched_speaker_id: Optional[int] = None
    matched_speaker_name: Optional[str] = None

    def summary(self) -> Dict[str, Optional[str]]:
        return {
            "speaker_label": self.speaker_label,
            "matched_speaker_id": self.matched_speaker_id,
            "matched_speaker_name": self.matched_speaker_name,
            "inferred_names": self.inferred_names,
            "features": self.features,
            "stats": self.stats,
            "psychometrics": self.psychometrics,
        }


@dataclass
class SpeakerMatch:
    speaker_profile: SpeakerProfile
    score: float
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None


@dataclass
class ProcessedMeeting:
    turns: List[SpeakerTurn]
    profiles: List[SpeakerProfile]
    transcript_path: Optional[str] = None
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, List[Dict[str, Optional[str]]]]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "profiles": [profile.summary() for profile in self.profiles],
            "transcript_path": self.transcript_path,
            "report_path": self.report_path,
        }
