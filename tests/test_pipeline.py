from __future__ import annotations

import json
from pathlib import Path

from speakerai.models import SpeakerProfile, SpeakerTurn
from speakerai.pipeline import SpeakerPipeline


class DummyTranscriber:
    def transcribe(self, audio_path):
        return [
            SpeakerTurn(speaker_label="Speaker A", start=0.0, end=5.0, text="My name is Alice"),
            SpeakerTurn(speaker_label="Speaker B", start=5.0, end=9.0, text="Project update"),
            SpeakerTurn(speaker_label="Speaker A", start=9.0, end=12.0, text="We are happy"),
        ]


class DummyFeatureExtractor:
    def extract_features(self, audio_path, texts, start, end):
        from speakerai.audio.features import FeatureSet

        return FeatureSet(
            mean_pitch=200.0,
            pitch_std=5.0,
            speaking_rate=150.0,
            energy_mean=0.5,
            energy_std=0.1,
            spectral_centroid_mean=1000.0,
            spectral_centroid_std=50.0,
        )


class DummyPsychometrics:
    def analyze(self, texts, stats):
        return {"excited": 1.0, "happy": 2.0, "valence": 0.5, "arousal": 0.2, "dominance": 0.1, "angry": 0.0, "sad": 0.0, "fun": 1.0}


class DummyDatabase:
    def __init__(self):
        self.saved = []

    def get_all_speakers(self):
        return []

    def add_speaker(self, name, features, stats, psychometrics, description=None):
        self.saved.append((name, features, stats, psychometrics))
        return 1

    def update_speaker(self, *args, **kwargs):
        pass


def test_pipeline_creates_outputs(tmp_path: Path, monkeypatch):
    pipeline = SpeakerPipeline(
        transcription=DummyTranscriber(),
        feature_extractor=DummyFeatureExtractor(),
        psychometrics=DummyPsychometrics(),
        database=DummyDatabase(),
    )
    audio_file = tmp_path / "meeting.mp3"
    audio_file.write_bytes(b"fake audio")

    result = pipeline.process(audio_file)

    assert result.transcript_path
    assert result.report_path

    transcript_file = Path(result.transcript_path)
    report_file = Path(result.report_path)

    assert transcript_file.exists()
    assert report_file.exists()

    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data[0]["speaker_label"] == "Speaker A"
    assert "psychometrics" in data[0]
