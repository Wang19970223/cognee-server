# AGENTS.md — Guidelines for AI Coding Assistants

## Project Overview

**Cognee Server** — FastAPI backend wrapping the Cognee knowledge graph framework with Celery task queue support.
Python 3.10+, virtual environment `.venv/`.

## Repository Structure

```
app/
  main.py              # Entrypoint — patches applied BEFORE `import cognee`
  config.py            # Pydantic Settings (.env)
  cognee_patches.py    # Runtime patches for litellm/tiktoken/proxy
  auth.py              # API key auth (read/write/admin hierarchy)
  api/                 # Route definitions
  schemas/             # Pydantic request/response models
  tasks/               # Celery task definitions
celery_worker.py       # Worker entrypoint
```

## Build, Run, and Test

```bash
# Activate environment
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate         # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Start API server
python -m app.main

# Start Celery worker (separate terminal)
celery -A celery_worker worker --loglevel=info
```

**Test commands:** No test suite exists yet. Create one under `tests/` using `pytest` when adding testable logic.
```bash
pip install pytest httpx
pytest tests/
```

## Critical Import Ordering

`app/cognee_patches.py` must be imported and applied **before** any `import cognee` statement globally:

```python
import app.cognee_patches  # noqa: E402 — MUST be first
app.cognee_patches.apply()

import cognee  # safe now
```

Patches handle:
- DashScope `litellm` incompatibilities
- JSON parsing bugs in `qwen-max`
- Corporate proxy/SSL issues for `tiktoken` downloads

If you add a new module that imports `cognee`, ensure the patch is already applied at startup.

## Code Style

### File Headers
Every Python file should start with:
```python
from __future__ import annotations
```

### Imports
- Standard library → third-party → local application (grouped with blank lines)
- Use `# noqa: E402` only for deferred imports required by the patching lifecycle
- Absolute imports: `from app.config import settings`

### Formatting (no linter configured)
- 4-space indentation, no tabs
- Black-style line length (~88 chars) — follow what you see in the codebase
- Trailing commas on multiline dicts/lists

### Naming
- `snake_case` for functions, variables, modules
- `PascalCase` for classes and Pydantic models
- `UPPER_CASE` for constants
- `settings` (singleton) from `app/config.py` for all config access

### Types
- Use type hints on all function signatures — this codebase is typed
- Leverage `pydantic.BaseModel` for request/response schemas
- Use `Annotated[...]` with FastAPI `Query()`/`Path()` for route parameters

### Error Handling
- HTTP errors: raise `fastapi.HTTPException(status_code=..., detail=...)`
- Validation errors: let Pydantic handle automatically (422)
- Task errors: Celery retries via `self.retry(exc=...)` with exponential backoff
- Log with `structlog.get_logger()` — no `print()` statements

### Auth
Routes use FastAPI dependency injection:
```python
async def list_graphs(_user = Depends(require_write)):
```
Levels: `require_read` | `require_write` | `require_admin`

## Configuration

All config from `.env` via Pydantic Settings in `app/config.py`:
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- `API_KEYS` (comma-separated)
- Cognee-specific: `COGNEE_DB_PATH`, `LLM_MODEL`, `LLM_PROVIDER`, etc.

Never hardcode credentials. Never commit `.env` or `env` files.

## Environment Files (DO NOT COMMIT)

Ignored: `.env`, `.env.local`, `.env.*`, `env`, `.venv/`, `.idea/`

Create your own `.env` by copying a template if one exists, or reference `app/config.py` for required keys.

## Commit Conventions

- Use imperative mood: "add", "fix", "refactor", not "added" or "adding"
- Keep messages concise, 1-2 lines
- Scope optional: `fix(auth): handle expired API keys`
- Group related changes; don't mix unrelated features in one commit

## When Asked to "Write Tests"

Default to `pytest`:
```
tests/
  __init__.py
  test_api_graphs.py
  test_auth.py
```
Use `httpx.AsyncClient` with `ASGITransport` for FastAPI integration tests.
Mark slow tests with `@pytest.mark.slow`.
