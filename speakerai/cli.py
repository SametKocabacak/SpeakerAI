"""Command line interface for SpeakerAI."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import SpeakerPipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process meeting audio files to transcripts and speaker reports.")
    parser.add_argument("audio", type=str, help="Path to the meeting audio file (mp3).")
    parser.add_argument("--team", nargs="*", default=None, help="Optional list of team member names for matching.")
    parser.add_argument("--log", default="INFO", help="Logging level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log.upper(), format="%(levelname)s %(name)s: %(message)s")
    pipeline = SpeakerPipeline()
    meeting = pipeline.process(Path(args.audio), team_members=args.team)
    LOGGER.info("Transcript saved to %s", meeting.transcript_path)
    LOGGER.info("Speaker report saved to %s", meeting.report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
