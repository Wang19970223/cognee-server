"""
app/api/search.py
─────────────────────────────────────────────────────────────────────────────
POST /api/v1/search         – natural language knowledge retrieval
POST /api/v1/search/similar – similarity-based retrieval (same backend,
                              reuses /search logic with the provided text)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import require_read
from app.config import get_settings
from app.schemas.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarSearchRequest,
    SourceInfo,
)

router = APIRouter()


@router.post("/search", response_model=SearchResponse, summary="Natural language knowledge retrieval")
async def search(
    body: SearchRequest,
    _perm: str = Depends(require_read),
):
    return await _do_search(
        query=body.query,
        context_type=body.context_type,
        max_results=body.max_results,
        search_type=body.search_type.value,
    )


@router.post(
    "/search/similar",
    response_model=SearchResponse,
    summary="Similarity-based retrieval",
)
async def search_similar(
    body: SimilarSearchRequest,
    _perm: str = Depends(require_read),
):
    return await _do_search(
        query=body.text,
        context_type=body.context_type,
        max_results=body.max_results,
        search_type="TRIPLET_COMPLETION",
    )


# ── Internal ──────────────────────────────────────────────────────────────────

async def _do_search(
    query: str,
    context_type: list | None,
    max_results: int,
    search_type: str,
) -> SearchResponse:
    import cognee
    from cognee.modules.search.types.SearchType import SearchType

    settings = get_settings()
    datasets = context_type or [settings.cognee_dataset]
    request_id = str(uuid.uuid4())[:8]
    t_start = time.monotonic()

    stype = getattr(SearchType, search_type, SearchType.GRAPH_COMPLETION)

    try:
        raw_results: List[Any] = await cognee.search(
            query,
            query_type=stype,
            datasets=datasets[0] if len(datasets) == 1 else datasets,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cognee search error: {exc}",
        )

    elapsed_ms = int((time.monotonic() - t_start) * 1000)

    results = [
        _coerce_result(i, item, search_type)
        for i, item in enumerate(raw_results[:max_results])
    ]

    return SearchResponse(
        request_id=f"req_{request_id}",
        query=query,
        results=results,
        total=len(results),
        processing_time_ms=elapsed_ms,
    )


def _coerce_result(idx: int, item: Any, search_type: str) -> SearchResult:
    """Normalize cognee result (dict or object) into SearchResult."""
    if isinstance(item, dict):
        if search_type == "TRIPLET_COMPLETION":
            subj = item.get("subject") or item.get("node") or item.get("source", "")
            rel = (
                item.get("relationship")
                or item.get("relation")
                or item.get("predicate")
                or item.get("type", "")
            )
            obj = item.get("object") or item.get("value") or item.get("target", "")
            content = f"{subj} -[{rel}]-> {obj}".strip()
            title = str(subj)[:120] if subj else f"Triple {idx + 1}"
        else:
            answer = (
                item.get("search_result")
                or item.get("answer")
                or item.get("text")
                or item.get("content")
                or ""
            )
            if isinstance(answer, list):
                answer = "\n".join(str(a) for a in answer)
            content = str(answer)
            title = content[:120] if content else f"Result {idx + 1}"

        source_path = item.get("source") or item.get("file_path")
        return SearchResult(
            id=item.get("id") or f"result_{idx}",
            type=item.get("type", "knowledge"),
            title=title,
            content=content[:2000],
            relevance_score=item.get("score") or item.get("relevance_score"),
            source=SourceInfo(path=str(source_path)) if source_path else None,
        )
    else:
        text = str(item)[:2000]
        return SearchResult(
            id=f"result_{idx}",
            type="knowledge",
            title=text[:120],
            content=text,
        )
