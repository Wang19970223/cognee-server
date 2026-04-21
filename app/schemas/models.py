"""
app/schemas/models.py
─────────────────────────────────────────────────────────────────────────────
Pydantic request/response models for all API endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Search ────────────────────────────────────────────────────────────────────

class SearchType(str, Enum):
    TRIPLET_COMPLETION = "TRIPLET_COMPLETION"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    SUMMARIES = "SUMMARIES"
    CHUNKS = "CHUNKS"
    RAG_COMPLETION = "RAG_COMPLETION"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    context_type: Optional[List[str]] = Field(
        None, description="Dataset names to search (e.g. ['recipes', 'decisions'])"
    )
    max_results: int = Field(5, ge=1, le=50)
    search_type: SearchType = Field(SearchType.GRAPH_COMPLETION)
    filters: Optional[Dict[str, Any]] = None


class SourceInfo(BaseModel):
    path: Optional[str] = None
    last_updated: Optional[str] = None


class EntityInfo(BaseModel):
    name: str
    type: str


class RelatedItem(BaseModel):
    id: str
    title: str
    relation: str


class SearchResult(BaseModel):
    id: str
    type: str
    title: str
    content: str
    relevance_score: Optional[float] = None
    source: Optional[SourceInfo] = None
    entities: List[EntityInfo] = []
    related: List[RelatedItem] = []


class SearchResponse(BaseModel):
    request_id: str
    query: str
    results: List[SearchResult]
    total: int
    processing_time_ms: int


class SimilarSearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    context_type: Optional[List[str]] = None
    max_results: int = Field(3, ge=1, le=50)


# ── Index ─────────────────────────────────────────────────────────────────────

class IndexMode(str, Enum):
    incremental = "incremental"
    full = "full"


class IndexRequest(BaseModel):
    mode: IndexMode = IndexMode.incremental
    urls: Optional[List[str]] = Field(
        None,
        description="List of HTTP/HTTPS URLs pointing to .md files to fetch and index",
    )
    data_sources: Optional[List[str]] = Field(
        None,
        description="List of local directory paths to scan for .md files (server-side)",
    )
    dataset: Optional[str] = Field(None, description="Target dataset name (overrides default)")
    force: bool = False


class IndexResponse(BaseModel):
    job_id: str
    status: str
    estimated_time_seconds: int
    check_url: str


# ── Jobs ──────────────────────────────────────────────────────────────────────

class JobStats(BaseModel):
    documents_processed: Optional[int] = None
    entities_extracted: Optional[int] = None
    relations_extracted: Optional[int] = None
    processing_time_ms: Optional[int] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | started | progress | success | failure | revoked
    progress: Optional[int] = None
    stats: Optional[JobStats] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ── Health ────────────────────────────────────────────────────────────────────

class DatasetHealth(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    last_updated: Optional[str] = None
    health_score: int = 0
    issues: List[str] = []


class HealthResponse(BaseModel):
    overall_score: int
    datasets: Dict[str, DatasetHealth]
    recommendations: List[str] = []


# ── Knowledge management ──────────────────────────────────────────────────────

class KnowledgeNodeRequest(BaseModel):
    type: str = Field(..., description="Entity type: recipe | decision | concept | code | bug")
    title: str
    content: str
    entities: Optional[List[EntityInfo]] = None
    relations: Optional[List[Dict[str, str]]] = None
    source_path: Optional[str] = None
    dataset: Optional[str] = None


class KnowledgeNodeResponse(BaseModel):
    id: str
    status: str
    message: str


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
