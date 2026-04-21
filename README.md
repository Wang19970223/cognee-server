# Cognee Knowledge Service

统一知识图谱服务，为 GitHub Copilot Skills/Agent 提供共享知识检索能力，避免每个成员重复搭建 Cognee 环境。

---

## 一、整体架构

```
调用方 (Skill / Agent / curl)
        │  X-API-Key + HTTP 请求
        ▼
┌─────────────────────────────────────┐
│   FastAPI  (uvicorn, port 8000)     │
│                                     │
│  /api/v1/search                     │  ← 同步，直接调 cognee.search()
│  /api/v1/search/similar             │  ← 同步
│  /api/v1/index     ─────────────────┼──► 派发 Celery Task，立即返回 job_id
│  /api/v1/jobs/{id}                  │  ← 查询 Redis 中的任务状态
│  /api/v1/health                     │  ← 同步健康度报告
│  /api/v1/knowledge  (POST / DELETE) │
└─────────────────────────────────────┘
        │ Celery 任务
        ▼
┌─────────────────────────────────────┐
│  Celery Worker (threads pool)       │
│                                     │
│  ingest_documents task:             │
│   1. 收集文档 (URL / 上传 / 本地)   │
│   2. cognee.add()                   │
│   3. cognee.cognify()  ◄────────────┼── 调用 DashScope LLM 提取实体关系
│   4. create_triplet_embeddings      │
└─────────────────────────────────────┘
        │ broker + result backend
        ▼
┌─────────────────┐
│   Redis         │  10.235.228.83:31111
│ 任务队列 + 结果  │
└─────────────────┘
```

---

## 二、数据存储（全部本地嵌入式，无需额外部署）

| 存储层 | 引擎 | 存储内容 | 默认路径 |
|--------|------|---------|---------|
| 关系型 DB | SQLite | 文档元数据、分块记录 | `.cognee_data/databases/cognee_db` |
| 图数据库 | KuzuDB（嵌入式） | 实体节点、关系边、知识三元组 | `.cognee_data/databases/graph_db/` |
| 向量数据库 | LanceDB（嵌入式） | Embedding 向量、相似度索引 | `.cognee_data/databases/vector_db/` |
| Tokenizer 缓存 | 本地文件 | BPE 词表（首次联网下载后缓存） | `~/.tiktoken/` |
| 任务队列 / 结果 | Redis | Celery 任务状态、进度、返回值 | `10.235.228.83:31111` |

> KuzuDB 和 LanceDB 均为**进程内嵌入式引擎**，数据直接写本地磁盘，无需启动任何数据库服务。

---

## 三、AI 模型配置

| 用途 | 模型 | 供应商 | 接入点 |
|------|------|--------|--------|
| LLM（实体抽取 / 图谱构建） | `qwen-plus` | 阿里云 DashScope | `dashscope.aliyuncs.com/compatible-mode/v1` |
| Embedding（向量化） | `text-embedding-v3` | 阿里云 DashScope | 同上 |
| Embedding 维度 | 1024 维 | — | — |
| Tokenizer | `cl100k_base`（tiktoken） | 本地缓存 | 首次从微软 Azure Blob 下载 |

> LLM 和 Embedding 均使用 OpenAI 兼容格式，通过 LiteLLM 路由；LiteLLM 的 embedding 调用已被直接替换为 HTTP 调用（绕过参数兼容性问题）。

---

## 四、认证与权限

在 `.env` 中配置：
```
COGNEE_API_KEYS=sk-team-read:read,sk-team-write:write,sk-admin:admin
```

通过请求头传递：`X-API-Key: <key>`

| 权限级别 | 允许操作 |
|---------|---------|
| `read`  | GET /health · POST /search · POST /search/similar · GET /jobs/{id} |
| `write` | 以上 + POST /index · POST /knowledge |
| `admin` | 以上 + DELETE /knowledge/{id} |

---

## 五、关键运行时补丁（app/cognee_patches.py）

以下补丁在 `import cognee` **之前**强制注入，解决企业网络 + DashScope 兼容性问题：

| 补丁 | 原因 |
|------|------|
| `litellm.aembedding` 替换 | DashScope 不支持 LiteLLM 某些参数；直接 HTTP 调用，支持批量并行（每批 ≤10 条） |
| `litellm.acompletion` 包装 | qwen-max function call 返回的 JSON 含未转义换行符，修复后 round-trip 规范化 |
| `tiktoken.encoding_for_model` 替换 | 将 `text-embedding-v3/v2` 映射到 `cl100k_base`，tiktoken 不认识 DashScope 模型名 |
| `tiktoken.load.read_file` 替换 | BPE 词表下载走企业代理 + 跳过 SSL 验证（自签名证书），仅首次下载触发 |

---

## 六、索引流程（为什么慢？）

```
输入文档
   │
   ▼
cognee.add()          ~秒级    分块存入 SQLite
   │
   ▼
cognee.cognify()      ← ⚠️ 最慢的步骤，见下方说明
   │
   ▼
create_triplet_embeddings()   向量化三元组 → LanceDB
```

### cognify 为什么慢？

`cognify()` 是**深度知识图谱构建**流程，对每个文档分块会依次调用多次 LLM：

```
每个分块（chunk）：
  ① LLM 调用：提取实体列表          ~2–5s
  ② LLM 调用：提取实体间关系         ~2–5s
  ③ LLM 调用：生成摘要               ~2–5s
  ④ LLM 调用：实体消歧/合并          ~2–5s（视实体数量）
```

一篇 ~1000 字的文档约被分为 3–5 个 chunk，每 chunk 约 4 次 LLM 调用，加上企业代理延迟：

| 文档大小 | 预估 LLM 调用次数 | 预估耗时 |
|---------|-----------------|---------|
| 小型（~500字） | 8–12 次 | 1–3 分钟 |
| 中型（~2000字） | 20–40 次 | 5–15 分钟 |
| 大型（~5000字） | 50–100 次 | 15–40 分钟 |

### 加速建议

| 方案 | 操作 | 效果 |
|------|------|------|
| 换更快模型 | `.env` 中改 `LLM_MODEL=openai/qwen-turbo` | 速度提升 2–3x，质量略降 |
| 减小文档大小 | 拆分大文档后再上传 | 线性减少调用次数 |
| 增加 Worker 并发 | `--concurrency=4` 但需注意 API 限流 | 多任务并行（单任务无效） |

---

## 七、API 接口一览

| Method | Path | 权限 | 说明 |
|--------|------|-----|------|
| GET  | `/api/v1/health` | read | 知识图谱健康度报告 |
| POST | `/api/v1/search` | read | 自然语言知识检索 |
| POST | `/api/v1/search/similar` | read | 相似性检索 |
| POST | `/api/v1/index` | write | 触发索引任务（立即返回 job_id） |
| GET  | `/api/v1/jobs/{job_id}` | read | 查询索引任务进度 |
| POST | `/api/v1/knowledge` | write | 手动添加知识节点 |
| DELETE | `/api/v1/knowledge/{id}` | admin | 删除知识节点 |

### /index 支持三种数据来源

**1. URL 列表（HTTP/HTTPS .md 文件）**
```json
{
  "mode": "incremental",
  "dataset": "knowledge",
  "urls": ["https://your-server/doc.md"]
}
```

**2. 文件上传（multipart）**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/index \
  -H "X-API-Key: sk-team-write" \
  -F "mode=incremental" \
  -F "files=@my-doc.md"
```

**3. 服务器本地路径**
```json
{
  "mode": "incremental",
  "data_sources": ["docs/design"]
}
```

### /search 支持的搜索类型

| search_type | 说明 | 推荐场景 |
|------------|------|---------|
| `GRAPH_COMPLETION` | 结合知识图谱的 RAG 回答（默认） | 自然语言问答 |
| `TRIPLET_COMPLETION` | 直接返回图谱三元组，不调 LLM | 验证图谱内容、调试 |
| `SUMMARIES` | 返回文档摘要节点 | 文档概览 |
| `CHUNKS` | 返回原始文本片段 | 原文检索 |

---

## 八、启动命令

```powershell
# 1. 激活虚拟环境（已有 .venv）
# 如已在 .venv 环境中则跳过

# 2. 启动 API 服务
cd C:\Users\SESA849562\workspace\cognee-server
.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000

# 3. 另开终端：启动 Celery Worker（索引任务执行者）
cd C:\Users\SESA849562\workspace\cognee-server
.venv\Scripts\celery.exe -A app.celery_app worker --loglevel=info --pool=threads --concurrency=2

# 4. 访问 API 文档（用 127.0.0.1，不要用 localhost——企业代理会拦截）
# http://127.0.0.1:8000/docs
```

---

## 九、环境变量说明

参见 `.env.example`，关键变量：

| 变量 | 说明 |
|------|------|
| `COGNEE_API_KEYS` | `key1:read,key2:write,key3:admin` |
| `REDIS_URL` | `redis://:password@host:port/db` |
| `LLM_API_KEY` | DashScope API Key |
| `LLM_MODEL` | 默认 `openai/qwen-plus`，可换 `openai/qwen-turbo` 提速 |
| `COGNEE_DATASET` | 默认数据集名称，默认 `knowledge` |
| `HTTP_PROXY` | 企业代理地址 |

---

## 十、Skill 集成示例

```python
import httpx

async def search_knowledge(query: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:8000/api/v1/search",
            headers={"X-API-Key": "sk-team-read"},
            json={
                "query": query,
                "max_results": 5,
                "search_type": "GRAPH_COMPLETION"
            },
        )
        resp.raise_for_status()
        return resp.json()
```
