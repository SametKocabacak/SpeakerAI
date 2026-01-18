"""Psychometric scoring based on acoustic and linguistic cues."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None

BASIC_EMOTION_KEYWORDS = {
    "excited": {"excited", "thrilled", "pumped", "energetic"},
    "angry": {"angry", "furious", "frustrated", "mad"},
    "happy": {"happy", "glad", "pleased", "delighted"},
    "sad": {"sad", "upset", "down", "unhappy"},
    "fun": {"fun", "enjoy", "laugh", "joke", "humor"},
}

DEFAULT_SCORES = {
    "excited": 0.0,
    "angry": 0.0,
    "happy": 0.0,
    "sad": 0.0,
    "fun": 0.0,
    "valence": 0.0,
    "arousal": 0.0,
    "dominance": 0.0,
}


class PsychometricAnalyzer:
    def __init__(self) -> None:
        self.sentiment = None
        if SentimentIntensityAnalyzer is not None:
            self.sentiment = SentimentIntensityAnalyzer()

    def analyze(self, texts: Iterable[str], acoustic_stats: Dict[str, float]) -> Dict[str, float]:
        combined_text = " ".join(texts)
        scores = DEFAULT_SCORES.copy()
        if self.sentiment is not None:
            sentiment_scores = self.sentiment.polarity_scores(combined_text)
            scores["valence"] = sentiment_scores.get("compound", 0.0)
            scores["arousal"] = sentiment_scores.get("pos", 0.0) - sentiment_scores.get("neg", 0.0)
            scores["dominance"] = sentiment_scores.get("pos", 0.0)
        else:
            LOGGER.warning("VADER sentiment analyzer not available; using keyword heuristics.")
            scores["valence"] = 0.0
        scores.update(self._keyword_scores(combined_text))
        scores = self._normalize(scores, acoustic_stats)
        return scores

    def _keyword_scores(self, combined_text: str) -> Dict[str, float]:
        word_counts = Counter(token.strip(".,!?").lower() for token in combined_text.split())
        keyword_scores = {}
        for metric, keywords in BASIC_EMOTION_KEYWORDS.items():
            count = sum(word_counts.get(keyword, 0) for keyword in keywords)
            keyword_scores[metric] = float(count)
        return keyword_scores

    def _normalize(self, scores: Dict[str, float], acoustic_stats: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        energy = acoustic_stats.get("energy_mean", 0.0)
        pitch = acoustic_stats.get("mean_pitch", 0.0)
        for key, value in scores.items():
            normalized[key] = float(value)
        normalized["arousal"] = float(scores.get("arousal", 0.0) + energy / 10.0)
        normalized["dominance"] = float(scores.get("dominance", 0.0) + pitch / 400.0)
        return normalized
