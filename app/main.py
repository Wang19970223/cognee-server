"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

Startup order (CRITICAL):
  1. Load settings (pydantic-settings reads .env)
  2. Call settings.apply_to_environ() → sets all env vars for cognee
  3. Apply litellm / tiktoken patches
  4. Import cognee (now picks up the patched env)
  5. Mount API routers

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Step 1: load settings & apply env vars BEFORE any cognee import ──────────
from app.config import get_settings

settings = get_settings()
settings.apply_to_environ()

# ── Step 2: apply patches (litellm + tiktoken) ───────────────────────────────
from app.cognee_patches import apply_patches  # noqa: E402

apply_patches(
    llm_api_key=settings.llm_api_key,
    llm_endpoint=settings.llm_endpoint,
    http_proxy=settings.http_proxy,
)

# ── Step 3: import cognee only after patches and env are in place ─────────────
import cognee  # noqa: E402, F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing async needed (cognee uses lazy init)
    yield
    # Shutdown: clean up any open connections
    try:
        pass
    except Exception:
        pass


app = FastAPI(
    title="Cognee Knowledge Service",
    description=(
        "Unified knowledge graph service for GitHub Copilot Skills and Agents. "
        "Provides natural-language knowledge retrieval, incremental indexing, "
        "and graph health monitoring via a REST API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS (adjust origins for production) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
from app.api import health, index, jobs, knowledge, search  # noqa: E402

PREFIX = "/api/v1"

app.include_router(search.router, prefix=PREFIX, tags=["Search"])
app.include_router(index.router, prefix=PREFIX, tags=["Index"])
app.include_router(jobs.router, prefix=PREFIX, tags=["Jobs"])
app.include_router(health.router, prefix=PREFIX, tags=["Health"])
app.include_router(knowledge.router, prefix=PREFIX, tags=["Knowledge"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Cognee Knowledge Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
