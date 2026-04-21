"""
app/api/health.py
─────────────────────────────────────────────────────────────────────────────
GET /api/v1/health  – Knowledge graph health report

Queries cognee with broad terms to estimate node/edge counts per dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import require_read
from app.config import get_settings
from app.schemas.models import DatasetHealth, HealthResponse

router = APIRouter()

_BROAD_TERMS = ["graph", "element", "symbol", "design", "module", "data", "service", "knowledge"]


@router.get("/health", response_model=HealthResponse, summary="Knowledge graph health report")
async def health(
    _perm: str = Depends(require_read),
):
    import cognee
    from cognee.modules.search.types.SearchType import SearchType

    settings = get_settings()
    datasets_to_check = [settings.cognee_dataset]

    dataset_health: Dict[str, DatasetHealth] = {}
    total_score = 0

    for dataset in datasets_to_check:
        seen = set()
        node_count = 0

        for term in _BROAD_TERMS:
            try:
                results = await cognee.search(
                    term,
                    query_type=SearchType.TRIPLET_COMPLETION,
                    datasets=dataset,
                )
                for item in results:
                    key = str(item)
                    if key not in seen:
                        seen.add(key)
                        node_count += 1
            except Exception:
                pass

        issues: List[str] = []
        if node_count == 0:
            issues.append("No indexed nodes found — run /index to populate the knowledge graph")

        score = min(100, node_count) if node_count > 0 else 0
        total_score += score

        dataset_health[dataset] = DatasetHealth(
            node_count=node_count,
            edge_count=0,
            last_updated=datetime.now(timezone.utc).isoformat(),
            health_score=score,
            issues=issues,
        )

    recommendations: List[str] = []
    for ds, dh in dataset_health.items():
        recommendations.extend(dh.issues)

    overall = total_score // max(len(datasets_to_check), 1)

    return HealthResponse(
        overall_score=overall,
        datasets=dataset_health,
        recommendations=recommendations,
    )
