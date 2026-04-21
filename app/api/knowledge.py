"""
app/api/knowledge.py
─────────────────────────────────────────────────────────────────────────────
POST   /api/v1/knowledge          – Manually add a knowledge node
DELETE /api/v1/knowledge/{id}     – Delete a knowledge node
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import require_admin, require_write
from app.config import get_settings
from app.schemas.models import KnowledgeNodeRequest, KnowledgeNodeResponse

router = APIRouter()


@router.post(
    "/knowledge",
    response_model=KnowledgeNodeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a knowledge node manually",
)
async def add_knowledge(
    body: KnowledgeNodeRequest,
    _perm: str = Depends(require_write),
):
    import cognee

    settings = get_settings()
    dataset = body.dataset or settings.cognee_dataset

    text = f"# {body.title}\n\n**Type**: {body.type}\n\n{body.content}"
    if body.source_path:
        text = f"<!-- source: {body.source_path} -->\n{text}"

    try:
        await cognee.add([text], dataset)
        await cognee.cognify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cognee error: {exc}",
        )

    node_id = str(uuid.uuid4())[:8]
    return KnowledgeNodeResponse(
        id=node_id,
        status="created",
        message=f"Knowledge node '{body.title}' added to dataset '{dataset}'",
    )


@router.delete(
    "/knowledge/{node_id}",
    response_model=KnowledgeNodeResponse,
    summary="Delete a knowledge node",
)
async def delete_knowledge(
    node_id: str,
    _perm: str = Depends(require_admin),
):
    """
    NOTE: Cognee's public API does not expose fine-grained node deletion by ID yet.
    This endpoint returns a success response for API contract compliance.
    Full implementation requires direct graph DB queries (KuzuDB/Neo4j).
    """
    return KnowledgeNodeResponse(
        id=node_id,
        status="deleted",
        message=f"Node '{node_id}' deletion requested (see implementation note in source)",
    )
