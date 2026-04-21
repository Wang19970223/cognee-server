# Cognee Knowledge Server

A personal knowledge graph service built with FastAPI and Cognee, providing knowledge retrieval capabilities for AI agents and skills.

---

## Architecture

```
Client (curl / agent / application)
        │  X-API-Key + HTTP request
        ▼
┌─────────────────────────────────────┐
│   FastAPI  (uvicorn, port 8000)     │
│                                     │
│  /api/v1/search                     │  ← Sync, direct cognee.search()
│  /api/v1/search/similar             │  ← Sync
│  /api/v1/index     ─────────────────┼──► Dispatches Celery Task, returns job_id
│  /api/v1/jobs/{id}                  │  ← Query task status via Redis
│  /api/v1/health                     │  ← Sync health report
│  /api/v1/knowledge  (POST / DELETE) │
└─────────────────────────────────────┘
        │ Celery tasks
        ▼
┌─────────────────────────────────────┐
│  Celery Worker (threads pool)       │
│                                     │
│  ingest_documents task:             │
│   1. Collect documents (URL/upload) │
│   2. cognee.add()                   │
│   3. cognee.cognify() ◄─────────────┼── Call LLM for entity/relation extraction
│   4. create_triplet_embeddings      │
└─────────────────────────────────────┘
        │ broker + result backend
        ▼
┌─────────────────┐
│   Redis         │
│ Queue + Result  │
└─────────────────┘
```

---

## Data Storage (All Local/Embedded)

| Layer | Engine | Content | Default Path |
|-------|--------|---------|--------------|
| Relational DB | SQLite | Document metadata, chunk records | `.cognee_data/databases/cognee_db` |
| Graph DB | KuzuDB (embedded) | Entity nodes, relationships, triplets | `.cognee_data/databases/graph_db/` |
| Vector DB | LanceDB (embedded) | Embeddings, similarity index | `.cognee_data/databases/vector_db/` |
| Tokenizer Cache | Local file | BPE vocabulary | `~/.tiktoken/` |
| Task Queue / Result | Redis | Celery task status, progress | Configured via `REDIS_URL` |

> KuzuDB and LanceDB are **in-process embedded engines**, data is written directly to local disk, no database services required.

---

## AI Model Configuration

| Purpose | Model | Provider | Endpoint |
|---------|-------|----------|----------|
| LLM (entity extraction / graph building) | `qwen-plus` | Alibaba DashScope | `dashscope.aliyuncs.com/compatible-mode/v1` |
| Embedding (vectorization) | `text-embedding-v3` | Alibaba DashScope | Same as above |
| Embedding Dimensions | 1024 | — | — |
| Tokenizer | `cl100k_base` (tiktoken) | Local cache | Downloaded on first run |

> LLM and Embedding use OpenAI-compatible format, routed through LiteLLM.

---

## Authentication

Configure in `.env`:
```
COGNEE_API_KEYS=sk-read:read,sk-write:write,sk-admin:admin
```

Pass via header: `X-API-Key: <key>`

| Permission | Allowed Operations |
|------------|-------------------|
| `read`  | GET /health · POST /search · POST /search/similar · GET /jobs/{id} |
| `write` | All read + POST /index · POST /knowledge |
| `admin` | All write + DELETE /knowledge/{id} |

---

## Runtime Patches (app/cognee_patches.py)

Patches injected before `import cognee` to handle compatibility issues:

| Patch | Reason |
|-------|--------|
| `litellm.aembedding` replaced | DashScope doesn't support some LiteLLM parameters; direct HTTP calls with batch parallel |
| `litellm.acompletion` wrapped | Fix unescaped newlines in qwen-max function call JSON responses |
| `tiktoken.encoding_for_model` replaced | Map DashScope model names to `cl100k_base` |
| `tiktoken.load.read_file` replaced | Handle BPE vocabulary download with proxy/SSL considerations |

---

## Indexing Pipeline (Why is it slow?)

```
Input Document
    │
    ▼
cognee.add()          ~seconds    Document chunking & storage
    │
    ▼
cognee.cognify()      ← ⚠️ Slowest step, see below
    │
    ▼
create_triplet_embeddings()   Vectorize triplets → LanceDB
```

### Why is cognify slow?

`cognify()` builds a knowledge graph by calling an LLM multiple times per chunk:

```
Per chunk:
  ① LLM: Extract entities           ~2–5s
  ② LLM: Extract relationships       ~2–5s
  ③ LLM: Generate summary            ~2–5s
  ④ LLM: Entity disambiguation       ~2–5s
```

| Document Size | Est. LLM Calls | Est. Time |
|---------------|----------------|-----------|
| Small (~500 chars) | 8–12 | 1–3 min |
| Medium (~2000 chars) | 20–40 | 5–15 min |
| Large (~5000 chars) | 50–100 | 15–40 min |

### Speed-up Tips

| Approach | How | Effect |
|----------|-----|--------|
| Use faster model | Set `LLM_MODEL=openai/qwen-turbo` in `.env` | 2–3x faster, slightly lower quality |
| Smaller documents | Split large docs before uploading | Linear reduction in calls |
| Increase worker concurrency | `--concurrency=4` (watch for rate limits) | Parallel task execution |

---

## API Reference

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET  | `/api/v1/health` | read | Knowledge graph health report |
| POST | `/api/v1/search` | read | Natural language knowledge search |
| POST | `/api/v1/search/similar` | read | Similarity search |
| POST | `/api/v1/index` | write | Trigger indexing task (returns job_id) |
| GET  | `/api/v1/jobs/{job_id}` | read | Query task progress |
| POST | `/api/v1/knowledge` | write | Manually add knowledge node |
| DELETE | `/api/v1/knowledge/{id}` | admin | Delete knowledge node |

### /index Data Sources

**1. URL List (HTTP/HTTPS .md files)**
```json
{
  "mode": "incremental",
  "dataset": "knowledge",
  "urls": ["https://example.com/doc.md"]
}
```

**2. File Upload (multipart)**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/index \
  -H "X-API-Key: sk-write" \
  -F "mode=incremental" \
  -F "files=@my-doc.md"
```

**3. Server Local Path**
```json
{
  "mode": "incremental",
  "data_sources": ["docs/design"]
}
```

### /search Search Types

| search_type | Description | Use Case |
|-------------|-------------|----------|
| `GRAPH_COMPLETION` | RAG with knowledge graph (default) | Natural language Q&A |
| `TRIPLET_COMPLETION` | Return raw triplets, no LLM call | Debug graph content |
| `SUMMARIES` | Return document summary nodes | Document overview |
| `CHUNKS` | Return raw text chunks | Raw text search |

---

## Quick Start

```bash
# 1. Create virtual environment (if not exists)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 4. Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. In another terminal, start Celery Worker
celery -A app.celery_app worker --loglevel=info --pool=threads --concurrency=2

# 6. Open API docs in browser
# http://127.0.0.1:8000/docs
```

---

## Environment Variables

See `.env.example` for full reference:

| Variable | Description |
|----------|-------------|
| `COGNEE_API_KEYS` | `key1:read,key2:write,key3:admin` |
| `REDIS_URL` | `redis://:password@host:port/db` |
| `LLM_API_KEY` | DashScope API Key |
| `LLM_MODEL` | Default `openai/qwen-plus`, use `openai/qwen-turbo` for speed |
| `COGNEE_DATASET` | Default dataset name, `knowledge` |
| `HTTP_PROXY` | HTTP proxy (if needed) |

---

## Integration Example

```python
import httpx

async def search_knowledge(query: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:8000/api/v1/search",
            headers={"X-API-Key": "sk-read"},
            json={
                "query": query,
                "max_results": 5,
                "search_type": "GRAPH_COMPLETION"
            },
        )
        resp.raise_for_status()
        return resp.json()
```
