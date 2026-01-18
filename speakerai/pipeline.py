"""Main orchestration pipeline."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from statistics import mean

try:  # pragma: no cover - optional dependency during tests
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from . import config
from .audio.features import FeatureExtractor
from .audio.psychometrics import PsychometricAnalyzer
from .audio.transcription import TranscriptionService
from .database import SpeakerDatabase
from .models import ProcessedMeeting, SpeakerProfile, SpeakerTurn

LOGGER = logging.getLogger(__name__)


class SpeakerPipeline:
    def __init__(
        self,
        transcription: Optional[TranscriptionService] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        psychometrics: Optional[PsychometricAnalyzer] = None,
        database: Optional[SpeakerDatabase] = None,
    ) -> None:
        self.transcription = transcription or TranscriptionService()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.psychometrics = psychometrics or PsychometricAnalyzer()
        self.database = database or SpeakerDatabase()

    def process(self, audio_path: str | Path, team_members: Optional[List[str]] = None) -> ProcessedMeeting:
        audio_path = Path(audio_path)
        turns = self.transcription.transcribe(audio_path)
        profiles = self._build_profiles(audio_path, turns, team_members)
        transcript_path = self._write_transcript(turns, audio_path)
        report_path = self._write_report(profiles, audio_path)
        return ProcessedMeeting(turns=turns, profiles=profiles, transcript_path=transcript_path, report_path=report_path)

    def _build_profiles(
        self,
        audio_path: Path,
        turns: List[SpeakerTurn],
        team_members: Optional[List[str]] = None,
    ) -> List[SpeakerProfile]:
        grouped: Dict[str, List[SpeakerTurn]] = defaultdict(list)
        for turn in turns:
            grouped[turn.speaker_label].append(turn)

        all_profiles: List[SpeakerProfile] = []
        for speaker_label, speaker_turns in grouped.items():
            texts = [turn.text for turn in speaker_turns]
            start = min(turn.start for turn in speaker_turns)
            end = max(turn.end for turn in speaker_turns)
            features = self.feature_extractor.extract_features(audio_path, texts, start, end)
            stats = {
                "total_duration": sum(turn.duration for turn in speaker_turns),
                "turn_count": len(speaker_turns),
                "avg_turn_duration": float(mean([turn.duration for turn in speaker_turns]) if speaker_turns else 0.0),
                "word_count": sum(len(turn.text.split()) for turn in speaker_turns),
            }
            psychometrics_scores = self.psychometrics.analyze(texts, features.to_dict())
            inferred_names = self._infer_names(speaker_turns)
            profile = SpeakerProfile(
                speaker_label=speaker_label,
                features=features.to_dict(),
                stats=stats,
                psychometrics=psychometrics_scores,
                inferred_names=inferred_names,
            )
            match = self._match_speaker(profile, team_members)
            if match:
                profile.matched_speaker_id = match.get("speaker_id")
                profile.matched_speaker_name = match.get("speaker_name")
                for turn in speaker_turns:
                    turn.speaker_name = profile.matched_speaker_name
            all_profiles.append(profile)
        return all_profiles

    def _match_speaker(self, profile: SpeakerProfile, team_members: Optional[List[str]]) -> Optional[Dict[str, object]]:
        known_speakers = self.database.get_all_speakers()
        if not known_speakers:
            return None
        vector = self._vectorize(profile.features)
        best_score = 0.0
        best_match: Optional[Dict[str, object]] = None
        for speaker_id, name, feature_vector, _, _ in known_speakers:
            if team_members and name not in team_members:
                continue
            db_vector = self._vectorize(feature_vector)
            score = self._cosine_similarity(vector, db_vector)
            if score > best_score and score >= config.FEATURE_SIMILARITY_THRESHOLD:
                best_score = score
                best_match = {
                    "speaker_id": speaker_id,
                    "speaker_name": name,
                    "score": score,
                }
        if best_match:
            LOGGER.info("Matched %s with score %.2f", best_match["speaker_name"], best_score)
        return best_match

    def _vectorize(self, feature_dict: Dict[str, float]):
        values = [float(value) for key, value in sorted(feature_dict.items())]
        if np is not None:
            return np.array(values)
        return values

    def _cosine_similarity(self, a, b) -> float:
        if np is not None:
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        # Fallback manual cosine similarity for lists
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _infer_names(self, turns: Iterable[SpeakerTurn]) -> List[str]:
        keywords = ["i am", "my name is", "this is"]
        inferred = []
        for turn in turns:
            lower = turn.text.lower()
            for keyword in keywords:
                if keyword in lower:
                    segment = lower.split(keyword, 1)[1].strip()
                    name = segment.split()[0]
                    if name and name not in inferred:
                        inferred.append(name.title())
        return inferred

    def _write_transcript(self, turns: List[SpeakerTurn], audio_path: Path) -> str:
        import csv

        output_file = config.TRANSCRIPT_DIR / f"{audio_path.stem}_transcript.csv"
        with output_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["speaker_label", "speaker_name", "start", "end", "text", "confidence"])
            writer.writeheader()
            for turn in turns:
                writer.writerow(
                    {
                        "speaker_label": turn.speaker_label,
                        "speaker_name": turn.speaker_name or "",
                        "start": f"{turn.start:.2f}",
                        "end": f"{turn.end:.2f}",
                        "text": turn.text,
                        "confidence": f"{(turn.confidence or 0.0):.2f}",
                    }
                )
        return str(output_file)

    def _write_report(self, profiles: List[SpeakerProfile], audio_path: Path) -> str:
        report_path = config.REPORT_DIR / f"{audio_path.stem}_speakers.json"
        payload = [profile.summary() for profile in profiles]
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return str(report_path)

    def save_profile(self, name: str, profile: SpeakerProfile, description: Optional[str] = None) -> int:
        return self.database.add_speaker(name, profile.features, profile.stats, profile.psychometrics, description=description)

    def update_profile(self, speaker_id: int, profile: SpeakerProfile, description: Optional[str] = None) -> None:
        self.database.update_speaker(speaker_id, profile.features, profile.stats, profile.psychometrics, description=description)

