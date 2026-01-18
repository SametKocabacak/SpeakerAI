"""FastAPI application exposing the SpeakerAI pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import config
from ..models import SpeakerProfile
from ..pipeline import SpeakerPipeline

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="SpeakerAI", description="Meeting transcription and speaker analytics")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

pipeline = SpeakerPipeline()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    team_members: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only MP3 files are supported")
    upload_path = config.UPLOAD_DIR / file.filename
    content = await file.read()
    upload_path.write_bytes(content)
    team = [member.strip() for member in (team_members or "").split(",") if member.strip()]
    meeting = pipeline.process(upload_path, team_members=team or None)
    context = {
        "request": request,
        "meeting": meeting,
        "team_members": team,
    }
    return templates.TemplateResponse("results.html", context)


@app.post("/confirm", response_class=HTMLResponse)
async def confirm_match(
    request: Request,
    speaker_label: str = Form(...),
    speaker_id: int = Form(...),
    confirm: str = Form(...),
):
    if confirm == "yes":
        LOGGER.info("Confirmed speaker %s for %s", speaker_id, speaker_label)
    return RedirectResponse(url="/", status_code=303)


@app.post("/register", response_class=HTMLResponse)
async def register_speaker(
    request: Request,
    speaker_label: str = Form(...),
    speaker_name: str = Form(...),
    transcript_path: str = Form(...),
    report_path: str = Form(...),
):
    report_file = Path(report_path)
    if not report_file.exists():
        raise HTTPException(status_code=400, detail="Report file not found for speaker registration")
    import json

    data = json.loads(report_file.read_text(encoding="utf-8"))
    profile_data = next((item for item in data if item["speaker_label"] == speaker_label), None)
    if profile_data is None:
        raise HTTPException(status_code=400, detail="Speaker profile not found in report")
    profile = SpeakerProfile(
        speaker_label=profile_data["speaker_label"],
        features=profile_data["features"],
        stats=profile_data["stats"],
        psychometrics=profile_data["psychometrics"],
        inferred_names=profile_data.get("inferred_names", []),
    )
    pipeline.save_profile(speaker_name, profile)
    LOGGER.info("Registered new speaker %s for label %s", speaker_name, speaker_label)
    return RedirectResponse(url="/", status_code=303)


@app.get("/download/transcript")
async def download_transcript(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    return FileResponse(path=file_path, filename=file_path.name, media_type="text/csv")


@app.get("/download/report")
async def download_report(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=file_path, filename=file_path.name, media_type="application/json")
