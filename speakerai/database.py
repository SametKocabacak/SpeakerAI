"""SQLite database helpers for speaker profiling."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    feature_vector TEXT NOT NULL,
    stats TEXT NOT NULL,
    psychometrics TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER IF NOT EXISTS speakers_updated_at
AFTER UPDATE ON speakers
BEGIN
    UPDATE speakers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""


class SpeakerDatabase:
    """Manage speaker profile persistence."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or config.DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def add_speaker(self, name: str, feature_vector: Dict[str, float], stats: Dict[str, float], psychometrics: Dict[str, float], description: Optional[str] = None) -> int:
        payload = (
            name,
            description,
            json.dumps(feature_vector),
            json.dumps(stats),
            json.dumps(psychometrics),
        )
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO speakers (name, description, feature_vector, stats, psychometrics)
                VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
            return cursor.lastrowid

    def update_speaker(self, speaker_id: int, feature_vector: Dict[str, float], stats: Dict[str, float], psychometrics: Dict[str, float], description: Optional[str] = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE speakers
                SET feature_vector = ?, stats = ?, psychometrics = ?, description = ?
                WHERE id = ?
                """,
                (
                    json.dumps(feature_vector),
                    json.dumps(stats),
                    json.dumps(psychometrics),
                    description,
                    speaker_id,
                ),
            )
            conn.commit()

    def get_all_speakers(self) -> List[Tuple[int, str, Dict[str, float], Dict[str, float], Dict[str, float]]]:
        with self.connect() as conn:
            cursor = conn.execute("SELECT id, name, feature_vector, stats, psychometrics FROM speakers")
            results = []
            for speaker_id, name, vector_json, stats_json, psych_json in cursor.fetchall():
                results.append(
                    (
                        speaker_id,
                        name,
                        json.loads(vector_json),
                        json.loads(stats_json),
                        json.loads(psych_json),
                    )
                )
            return results

    def find_by_name(self, name: str) -> Optional[Tuple[int, str, Dict[str, float]]]:
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT id, name, feature_vector FROM speakers WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return row[0], row[1], json.loads(row[2])

