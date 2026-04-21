"""
app/tasks/index_tasks.py
─────────────────────────────────────────────────────────────────────────────
Celery task that ingests documents into the Cognee knowledge graph.

Supports two input modes:
  1. data_sources (list of local directory paths)  → scans for .md files
  2. uploaded_files (list of {filename, content})  → content provided directly

Flow mirrors cognee-ingest-docs.py:
  cognee.add → cognee.cognify → create_triplet_embeddings

State updates (stored in Celery result backend):
  { "status": "...", "progress": 0-100, "stats": {...} }
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Task
from celery.utils.log import get_task_logger

from app.celery_app import celery_app
from app.config import get_settings

logger = get_task_logger(__name__)


class _CogneeTask(Task):
    """Base task: applies patches and cognee settings once per worker process."""
    _patches_applied: bool = False

    def __call__(self, *args, **kwargs):
        if not _CogneeTask._patches_applied:
            settings = get_settings()
            settings.apply_to_environ()
            from app.cognee_patches import apply_patches
            apply_patches(
                llm_api_key=settings.llm_api_key,
                llm_endpoint=settings.llm_endpoint,
                http_proxy=settings.http_proxy,
            )
            _CogneeTask._patches_applied = True
        return super().__call__(*args, **kwargs)


@celery_app.task(
    bind=True,
    base=_CogneeTask,
    name="app.tasks.index_tasks.ingest_documents",
    max_retries=1,
    default_retry_delay=10,
)
def ingest_documents(
    self: Task,
    dataset: str,
    data_sources: Optional[List[str]] = None,
    uploaded_files: Optional[List[Dict[str, str]]] = None,
    urls: Optional[List[str]] = None,
    mode: str = "incremental",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Main indexing task.

    Args:
        dataset: Target cognee dataset name.
        data_sources: Local directory paths to scan for .md files.
        uploaded_files: List of {"filename": ..., "content": ...} dicts.
        urls: HTTP/HTTPS URLs pointing to .md files to fetch and index.
        mode: "incremental" or "full" (full triggers prune first).
        force: If True, always re-index even if data appears up-to-date.

    Returns:
        Stats dict: {documents_processed, entities_extracted, processing_time_ms}
    """
    start_time = time.time()

    def _safe_update(progress: int, step: str) -> None:
        """Update task state, swallowing Redis errors so a connection drop
        doesn't abort the whole indexing job."""
        try:
            self.update_state(state="PROGRESS", meta={"progress": progress, "step": step})
        except Exception as _e:
            logger.warning("update_state failed (Redis disconnect?): %s", _e)

    _safe_update(0, "collecting documents")

    docs = _collect_docs(data_sources or [], uploaded_files or [], urls or [])
    if not docs:
        return {"error": "No documents found", "documents_processed": 0}

    logger.info(
        "=== ingest_documents START | dataset=%s | docs=%d | mode=%s ===",
        dataset, len(docs), mode,
    )
    for name, content in docs:
        logger.info("  doc: %s  (%d chars)", name, len(content))

    _safe_update(10, f"collected {len(docs)} documents")

    try:
        result = asyncio.run(_run_ingest(_safe_update, dataset, docs, mode))
        elapsed = int((time.time() - start_time) * 1000)
        result["processing_time_ms"] = elapsed
        return result
    except Exception as exc:
        logger.error("Ingest failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_docs(
    data_sources: List[str],
    uploaded_files: List[Dict[str, str]],
    urls: List[str],
) -> List[tuple[str, str]]:
    """Return list of (relative_path_or_name, content) tuples."""
    import httpx
    docs: List[tuple[str, str]] = []

    # ── 1. Local directory paths ──────────────────────────────────────────────
    for source_path in data_sources:
        base = Path(source_path)
        if not base.is_absolute():
            base = Path.cwd() / base
        if not base.exists():
            logger.warning("Data source path not found: %s", base)
            continue
        for f in sorted(base.rglob("*.md")):
            try:
                content = f.read_text(encoding="utf-8")
                rel = str(f.relative_to(Path.cwd()))
                docs.append((rel, content))
                logger.info("Collected: %s (%.1f KB)", rel, len(content.encode()) / 1024)
            except Exception as e:
                logger.warning("Skipping %s: %s", f, e)

    # ── 2. HTTP/HTTPS URLs ────────────────────────────────────────────────────
    settings = get_settings()
    proxy = settings.http_proxy or None
    for url in urls:
        try:
            client_kwargs: dict = {"timeout": 30, "follow_redirects": True}
            if proxy:
                client_kwargs["proxy"] = proxy
            with httpx.Client(**client_kwargs) as client:
                r = client.get(url)
                r.raise_for_status()
            content = r.text
            # Use the last path segment as the filename
            filename = url.rstrip("/").split("/")[-1]
            if not filename.endswith(".md"):
                filename += ".md"
            docs.append((filename, content))
            logger.info("Fetched URL: %s (%.1f KB)", url, len(content.encode()) / 1024)
        except Exception as e:
            logger.warning("Failed to fetch URL %s: %s", url, e)

    # ── 3. Uploaded file content ──────────────────────────────────────────────
    for uf in uploaded_files:
        fname = uf.get("filename", "uploaded.md")
        content = uf.get("content", "")
        if content:
            docs.append((fname, content))
            logger.info("Collected upload: %s", fname)

    return docs


async def _run_ingest(
    update: Any,
    dataset: str,
    docs: List[tuple[str, str]],
    mode: str,
) -> Dict[str, Any]:
    import cognee
    import logging as _logging
    import time as _time

    # Temporarily enable INFO-level logging for cognee internals so we can
    # see what the LLM pipeline is doing.
    for _log_name in ("cognee", "litellm", "root"):
        _logging.getLogger(_log_name).setLevel(_logging.INFO)

    def _ts() -> str:
        return _time.strftime("%H:%M:%S")

    # Prepend source comment + title to each doc for better graph tracing
    texts = [
        f"<!-- source: {name} -->\n# {Path(name).stem}\n\n{content}"
        for name, content in docs
    ]

    if mode == "full":
        logger.info("[%s] [1/4] Pruning old data...", _ts())
        update(5, "pruning old data")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        logger.info("[%s] [1/4] Prune complete.", _ts())

    logger.info("[%s] [2/4] cognee.add() — storing %d document(s) into dataset '%s'...", _ts(), len(texts), dataset)
    update(20, "adding documents")
    t0 = _time.monotonic()
    await cognee.add(texts, dataset)
    logger.info("[%s] [2/4] cognee.add() done in %.1fs.", _ts(), _time.monotonic() - t0)

    logger.info(
        "[%s] [3/4] cognee.cognify() — building knowledge graph (LLM entity/relation extraction)...",
        _ts(),
    )
    logger.info(
        "[%s]        ⚠ This step calls the LLM multiple times per chunk. Expect 2–15 min depending on document size.",
        _ts(),
    )
    update(40, "building knowledge graph (cognify)")
    t0 = _time.monotonic()
    await cognee.cognify()
    elapsed = _time.monotonic() - t0
    logger.info("[%s] [3/4] cognee.cognify() done in %.1fs (%.1f min).", _ts(), elapsed, elapsed / 60)

    logger.info("[%s] [4/4] create_triplet_embeddings() — vectorising triplets...", _ts())
    update(80, "creating triplet embeddings")
    t0 = _time.monotonic()
    try:
        from cognee.modules.users.methods import get_default_user
        from cognee.memify_pipelines.create_triplet_embeddings import create_triplet_embeddings
        user = await get_default_user()
        # 15-min cap: with many stale triplets from prior failed runs this
        # step can hang; a timeout lets the task still complete and verify.
        await asyncio.wait_for(
            create_triplet_embeddings(user=user, dataset=dataset),
            timeout=900,
        )
        logger.info("[%s] [4/4] create_triplet_embeddings done in %.1fs.", _ts(), _time.monotonic() - t0)
    except asyncio.TimeoutError:
        logger.warning("[%s] [4/4] create_triplet_embeddings timed out after 15 min — "
                       "skipping. Use mode=full to prune stale graph data.", _ts())
    except Exception as e:
        logger.warning("[%s] [4/4] create_triplet_embeddings skipped: %s", _ts(), e)

    update(95, "verifying")
    logger.info("[%s] Verifying — running test search...", _ts())

    # Quick verification: count accessible triples
    entity_count = 0
    try:
        from cognee.modules.search.types.SearchType import SearchType
        results = await asyncio.wait_for(
            cognee.search(
                "knowledge graph entity",
                query_type=SearchType.TRIPLET_COMPLETION,
                datasets=dataset,
            ),
            timeout=60,
        )
        entity_count = len(results)
        logger.info("[%s] Verification search returned %d result(s).", _ts(), entity_count)
    except asyncio.TimeoutError:
        logger.warning("[%s] Verification search timed out — search may still work after reindex.", _ts())
    except Exception as _e:
        logger.warning("[%s] Verification search failed: %s", _ts(), _e)

    return {
        "documents_processed": len(docs),
        "entities_extracted": entity_count,
        "relations_extracted": 0,
    }
