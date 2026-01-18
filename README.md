# SpeakerAI
End-to-end speaker diarization, transcription, and longitudinal voice analysis framework using Whisper and acoustic feature modeling.

---

# SpeakerAI: An Automated Speaker Analysis Framework

SpeakerAI is an end-to-end speaker analysis framework designed to process meeting and conversation recordings into structured, speaker-aware transcripts and analytical reports.
The system integrates speech recognition, speaker diarization, acoustic feature extraction, and psychometric analysis to identify and characterize speakers across multiple sessions.

The primary objective of SpeakerAI is to enable persistent speaker recognition and longitudinal voice analysis through reusable speaker profiles.

---

## System Capabilities

* **Automatic Speech Recognition**
  High-accuracy transcription powered by OpenAI Whisper models.

* **Speaker Diarization**
  Segmentation and labeling of individual speakers within multi-speaker audio recordings.

* **Acoustic Feature Extraction**
  Extraction of pitch, speaking rate, energy, and spectral statistics to form long-term voice signatures.

* **Psychometric and Emotional Scoring**
  Quantitative analysis of emotional tone at the level of individual speaker turns.

* **Persistent Speaker Profiles**
  SQLite-based storage for speaker embeddings and acoustic signatures, enabling future speaker matching.

* **Dual Interaction Modes**

  * Web-based interface implemented with FastAPI
  * Command-line interface for batch processing

* **Structured Outputs**

  * Detailed, speaker-labeled transcripts
  * Speaker intelligence and analytics reports

---

## Automated Environment Setup

An automated bootstrap script is provided to create a virtual environment, install dependencies, and retrieve an FFmpeg binary when not available on the host system.

```bash
./scripts/bootstrap.sh
```

After successful execution, activate the virtual environment:

```bash
source .venv/bin/activate
```

---

## Manual Installation (Optional)

Ensure that Python 3.10 or later is installed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

Audio decoding relies on FFmpeg.
A compatible binary is automatically provided via `imageio-ffmpeg`, or FFmpeg may be installed through the system package manager.

---

## Running the Web Application

Start the FastAPI server:

```bash
uvicorn speakerai.web.app:app --host 0.0.0.0 --port 7883 --reload
```

Access the application at:

```
http://localhost:7883
```

Users may upload an audio file, review generated transcripts, confirm speaker matches, and export analysis artifacts.

---

## Command-Line Interface Usage

To process an audio recording via the CLI:

```bash
python -m speakerai.cli /path/to/meeting.mp3 --team "Alice Example" "Bob Example"
```

Generated outputs are stored in:

* `data/outputs/transcripts/`
* `data/outputs/reports/`

Speaker confirmation prompts appear when prior matches exist, and confirmed results update the SQLite database.

---

## Testing

Execute the full test suite using:

```bash
pytest
```

---

## Project Structure

```
speakerai/
├── audio/          # Transcription, diarization, and acoustic analysis modules
├── pipeline.py     # End-to-end orchestration logic
├── web/            # FastAPI application and UI assets
├── database.py     # SQLite persistence layer
│
data/
├── outputs/        # Generated transcripts and reports
│
tests/
├── Automated pipeline tests
```

---

## Troubleshooting

* **Model Downloads**
  The selected Whisper model (default: `small`) is downloaded during the first execution.

* **GPU Acceleration**
  CUDA-enabled PyTorch installations are recommended for GPU inference.

* **FFmpeg Issues**
  Re-run the bootstrap script or install FFmpeg via the operating system package manager if audio decoding fails.

---

## Intended Use Cases

* Meeting and interview analysis
* Longitudinal speaker identification
* Audio-based behavioral research
* Speech analytics and conversational AI research

---
