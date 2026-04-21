"""
app/api/index.py
─────────────────────────────────────────────────────────────────────────────
POST /api/v1/index
  – Accept JSON body (data_sources: list of local paths)
  – OR multipart form with uploaded .md files

Dispatches to the Celery `ingest_documents` task and returns job_id immediately.
"""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.auth import require_write
from app.config import get_settings
from app.schemas.models import IndexRequest, IndexResponse
from app.tasks.index_tasks import ingest_documents

router = APIRouter()


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Trigger knowledge indexing",
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_index(
    request: Request,
    _perm: str = Depends(require_write),
):
    """
    Accepts either:
    - JSON body: `{"mode": "incremental", "data_sources": ["/path/to/docs"], "dataset": "..."}`
    - Multipart form: `mode`, `dataset`, and one or more `files` (.md uploads)
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        return await _handle_upload(request)
    else:
        body_bytes = await request.body()
        try:
            body = IndexRequest.model_validate_json(body_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid request body: {exc}",
            )
        return _dispatch_task(
            mode=body.mode.value,
            dataset=body.dataset,
            data_sources=body.data_sources or [],
            uploaded_files=[],
            urls=body.urls or [],
            force=body.force,
        )


async def _handle_upload(request: Request) -> IndexResponse:
    form = await request.form()
    mode = form.get("mode", "incremental")
    dataset = form.get("dataset") or None
    force = str(form.get("force", "false")).lower() == "true"

    uploaded_files = []
    for key, value in form.multi_items():
        if key == "files" and hasattr(value, "read"):
            content = await value.read()
            uploaded_files.append({
                "filename": value.filename or "upload.md",
                "content": content.decode("utf-8", errors="replace"),
            })

    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Multipart request must include at least one .md file under the 'files' field",
        )

    return _dispatch_task(
        mode=mode,
        dataset=dataset,
        data_sources=[],
        uploaded_files=uploaded_files,
        urls=[],
        force=force,
    )


def _dispatch_task(
    mode: str,
    dataset: Optional[str],
    data_sources: List[str],
    uploaded_files: List[dict],
    urls: List[str],
    force: bool,
) -> IndexResponse:
    settings = get_settings()
    resolved_dataset = dataset or settings.cognee_dataset

    task = ingest_documents.delay(
        dataset=resolved_dataset,
        data_sources=data_sources,
        uploaded_files=uploaded_files,
        urls=urls,
        mode=mode,
        force=force,
    )

    return IndexResponse(
        job_id=task.id,
        status="started",
        estimated_time_seconds=120,
        check_url=f"/api/v1/jobs/{task.id}",
    )
