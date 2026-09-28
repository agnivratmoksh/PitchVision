from pathlib import Path
import threading
import traceback

from fastapi import FastAPI
from pydantic import BaseModel

from pipeline.pipeline import run_pipeline


app = FastAPI(
    title="PitchVision Inference API"
)

JOBS = {}


class AnalyzeRequest(BaseModel):
    job_id: str
    file_path: str


# ==================================================
# JOB UPDATE
# ==================================================

def update_job(
    job_id,
    progress,
    message,
    stage
):
    JOBS[job_id] = {
        **JOBS.get(job_id, {}),
        "status": "processing",
        "progress": progress,
        "message": message,
        "stage": stage
    }


# ==================================================
# WORKER
# ==================================================

def worker(
    job_id: str,
    file_path: str
):
    try:
        result = run_pipeline(
            job_id,
            file_path,
            lambda p, m, s: update_job(job_id, p, m, s)
        )

        JOBS[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Analysis complete.",
            "stage": "Result aggregation",
            "result": result
        }

    except Exception as exc:
        traceback.print_exc()

        JOBS[job_id] = {
            "status": "failed",
            "progress": 0,
            "message": "Analysis failed.",
            "stage": "Error",
            "error": str(exc)
        }


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "pitchvision-python"
    }


# ==================================================
# START ANALYSIS
# ==================================================

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not Path(request.file_path).exists():
        JOBS[request.job_id] = {
            "status": "failed",
            "progress": 0,
            "message": "Input file does not exist.",
            "stage": "Input validation",
            "error": request.file_path
        }

        return {
            "job_id": request.job_id,
            "accepted": False
        }

    JOBS[request.job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued.",
        "stage": "Python pipeline"
    }

    thread = threading.Thread(
        target=worker,
        args=(request.job_id, request.file_path),
        daemon=True
    )
    thread.start()

    return {
        "job_id": request.job_id,
        "accepted": True
    }


# ==================================================
# JOB STATUS
# ==================================================

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return JOBS.get(
        job_id,
        {
            "status": "not_found",
            "message": "Job not found."
        }
    )