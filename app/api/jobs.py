"""
app/api/jobs.py
─────────────────────────────────────────────────────────────────────────────
GET /api/v1/jobs/{job_id}  – Query Celery task status via result backend
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from celery.result import AsyncResult

from app.auth import require_read
from app.celery_app import celery_app
from app.schemas.models import JobStats, JobStatus

router = APIRouter()

# Celery state → canonical status string mapping
_STATE_MAP = {
    "PENDING": "pending",
    "RECEIVED": "pending",
    "STARTED": "started",
    "PROGRESS": "in_progress",
    "SUCCESS": "completed",
    "FAILURE": "failed",
    "REVOKED": "revoked",
    "RETRY": "retrying",
}


@router.get("/jobs/{job_id}", response_model=JobStatus, summary="Check indexing job status")
async def get_job(
    job_id: str,
    _perm: str = Depends(require_read),
):
    result: AsyncResult = celery_app.AsyncResult(job_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    state = result.state
    canonical = _STATE_MAP.get(state, state.lower())

    meta = result.info or {}

    # PROGRESS state carries {progress, step} in meta
    progress = None
    if state == "PROGRESS" and isinstance(meta, dict):
        progress = meta.get("progress")
    elif state == "SUCCESS":
        progress = 100

    stats: JobStats | None = None
    error: str | None = None
    res_dict = None

    if state == "SUCCESS" and isinstance(meta, dict):
        stats = JobStats(
            documents_processed=meta.get("documents_processed"),
            entities_extracted=meta.get("entities_extracted"),
            relations_extracted=meta.get("relations_extracted"),
            processing_time_ms=meta.get("processing_time_ms"),
        )
        res_dict = meta
    elif state == "FAILURE":
        error = str(meta) if not isinstance(meta, dict) else str(meta.get("exc_message", meta))

    return JobStatus(
        job_id=job_id,
        status=canonical,
        progress=progress,
        stats=stats,
        error=error,
        result=res_dict,
    )
