"""
cognee-search.py
────────────────────────────────────────────────────────────────
查询已导入 Cognee 知识图谱的文档。
前提：已运行 cognee-ingest-docs.py 完成文档导入。

用法：
  # 交互模式（直接运行，按提示输入）
  python cognee-search.py

  # 单次查询
  python cognee-search.py "图元的拖拽创建流程是什么？"

  # 使用不同搜索类型
  python cognee-search.py "符号的数据结构" --type GRAPH_COMPLETION
────────────────────────────────────────────────────────────────
"""

import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# Windows 终端/管道下强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 1. 加载 .env（必须在 import cognee 之前）
load_dotenv()

PROXY        = os.environ.get("HTTP_PROXY", "")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT",
                   "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL    = os.environ.get("LLM_MODEL",      "openai/qwen-max")
EMBED_MODEL  = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-v3")
EMBED_DIMS   = os.environ.get("EMBEDDING_DIMENSIONS", "1024")

os.environ["LLM_PROVIDER"]          = "openai"
os.environ["LLM_MODEL"]             = LLM_MODEL
os.environ["LLM_ENDPOINT"]          = LLM_ENDPOINT
os.environ["EMBEDDING_PROVIDER"]    = "openai"
os.environ["EMBEDDING_MODEL"]       = EMBED_MODEL
os.environ["EMBEDDING_ENDPOINT"]    = LLM_ENDPOINT
os.environ["EMBEDDING_API_KEY"]     = LLM_API_KEY
os.environ.pop("EMBEDDING_DIMENSIONS", None)
os.environ["EMBEDDING_DIMENSIONS"]  = EMBED_DIMS
os.environ["COGNEE_SKIP_CONNECTION_TEST"]    = "true"
os.environ["TELEMETRY_DISABLED"]            = "1"
os.environ["LOG_LEVEL"]                     = "ERROR"
os.environ["COGNEE_LOG_LEVEL"]              = "ERROR"
# 关闭多用户访问控制，确保能访问所有已导入的数据
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

if PROXY:
    os.environ["HTTP_PROXY"]  = PROXY
    os.environ["HTTPS_PROXY"] = PROXY

# ── 2. Patch litellm.aembedding（同 ingest 脚本）
import litellm as _litellm
import httpx as _httpx


async def _safe_aembedding(*args, **kwargs):
    """
    绕过 litellm 直接调 DashScope。
    DashScope 单批最多 10 条。使用 asyncio.gather 并行分批，
    确保总耗时 ≈ 单批耗时，避免超过 cognee 内部 30s asyncio.wait_for 限制。
    """
    model      = kwargs.get("model", "").replace("openai/", "")
    api_base   = kwargs.get("api_base") or LLM_ENDPOINT
    api_key    = kwargs.get("api_key")  or LLM_API_KEY
    input_data = kwargs.get("input", [])
    if isinstance(input_data, str):
        input_data = [input_data]

    client_kwargs = {"timeout": 60}
    if PROXY:
        client_kwargs["proxy"] = PROXY

    BATCH_SIZE = 10
    batches = [input_data[i : i + BATCH_SIZE] for i in range(0, len(input_data), BATCH_SIZE)]

    async def fetch_batch(batch):
        async with _httpx.AsyncClient(**client_kwargs) as client:
            r = await client.post(
                f"{api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": batch},
            )
            if r.status_code != 200:
                raise RuntimeError(f"Embedding failed {r.status_code}: {r.text[:300]}")
            return r.json()["data"]

    import asyncio as _asyncio
    batch_results = await _asyncio.gather(*[fetch_batch(b) for b in batches])

    all_data = []
    offset = 0
    for items in batch_results:
        for item in items:
            item["index"] = offset + item.get("index", 0)
            all_data.append(item)
        offset += len(items)

    class _Resp:
        def __init__(self, data):
            self.data = data

    return _Resp(all_data)


_litellm.aembedding = _safe_aembedding

# ── 2b. Patch litellm.acompletion
#    qwen-max 的 function call arguments 里可能含未转义换行符（控制字符），
#    导致 instructor/Pydantic 的 JSON 解析失败。
#    修复：调用原始 acompletion 后，对每个 tool_call.function.arguments
#    做一次 json.loads(strict=False) + json.dumps 的 round-trip 规范化。
import json as _json

_orig_acompletion = _litellm.acompletion


def _fix_tool_args(arguments: str) -> str:
    try:
        return _json.dumps(_json.loads(arguments, strict=False), ensure_ascii=False)
    except Exception:
        return arguments


async def _safe_acompletion(*args, **kwargs):
    response = await _orig_acompletion(*args, **kwargs)
    try:
        for choice in response.choices or []:
            msg = getattr(choice, "message", None)
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn and getattr(fn, "arguments", None):
                    fn.arguments = _fix_tool_args(fn.arguments)
    except Exception:
        pass
    return response


_litellm.acompletion = _safe_acompletion

# ── 3. tiktoken 别名
import tiktoken as _tiktoken

_orig_enc = _tiktoken.encoding_for_model


def _patched_enc(model_name):
    _alias = {
        "text-embedding-v2": "cl100k_base",
        "text-embedding-v3": "cl100k_base",
    }
    return (
        _tiktoken.get_encoding(_alias[model_name])
        if model_name in _alias
        else _orig_enc(model_name)
    )


_tiktoken.encoding_for_model = _patched_enc

# ── 4. 现在才 import cognee
import cognee

# ── 支持的搜索类型（cognee SearchType）
# TRIPLET_COMPLETION - 返回图谱中实际提取的实体/关系三元组（最可靠，不依赖 LLM 补全）
# SUMMARIES         - 返回文档摘要节点
# CHUNKS            - 返回原始文本片段（需要 DocumentChunk_text 向量集合）
# GRAPH_COMPLETION  - 结合知识图谱的 RAG 回答（需要 LLM 补全，可能产生幻觉）
SEARCH_TYPES = ["TRIPLET_COMPLETION", "SUMMARIES", "CHUNKS", "GRAPH_COMPLETION", "RAG_COMPLETION"]

DATASET_NAME = os.environ.get("COGNEE_DATASET", "generic-graphic-ui-docs")


def fmt_result(idx: int, item, search_type: str = "") -> str:
    """格式化单条搜索结果，兼容字典和对象两种返回格式。"""
    lines = [f"\n[{idx}] ──────────────────────────────────────"]
    if isinstance(item, dict):
        if search_type == "TRIPLET_COMPLETION":
            # TRIPLET_COMPLETION 返回三元组：node1 -[关系]-> node2
            subj = item.get("subject", item.get("node", item.get("source", "")))
            rel  = item.get("relationship", item.get("relation", item.get("predicate", item.get("type", ""))))
            obj  = item.get("object", item.get("value", item.get("target", "")))
            if subj or rel or obj:
                lines.append(f"  {subj}  -[{rel}]->  {obj}")
                return "\n".join(lines)
        # 通用：search_result 字段通常是实际答案
        answer = item.get("search_result") or item.get("answer") or item.get("text") or item.get("content")
        if answer:
            if isinstance(answer, list):
                answer = "\n".join(str(a) for a in answer)
            lines.append(str(answer)[:800])
        else:
            # 兜底：打印整个 dict（去掉嵌入向量等大字段）
            clean = {k: v for k, v in item.items() if k not in ("embedding", "vector")}
            lines.append(str(clean)[:800])
    else:
        lines.append(str(item)[:800])
    return "\n".join(lines)


async def do_list():
    """列出图谱中所有已索引的实体节点，帮助诊断导入结果。"""
    print(f"\n数据集 '{DATASET_NAME}' 的图谱内容诊断")
    print("─" * 56)
    from cognee.modules.search.types.SearchType import SearchType

    # 用宽泛关键词做 TRIPLET_COMPLETION 搜索，收集图谱中所有可见节点/关系
    broad_terms = ["graph", "element", "symbol", "stage", "canvas", "design", "module", "data"]
    seen = set()
    all_triples = []

    for term in broad_terms:
        try:
            results = await cognee.search(term, query_type=SearchType.TRIPLET_COMPLETION, datasets=DATASET_NAME)
            for item in results:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(item)
        except Exception:
            pass

    if not all_triples:
        print("[空] 图谱中没有任何 TRIPLET_COMPLETION 数据。")
        print("     可能原因：")
        print("       1. cognify() 未完成，或中途出错")
        print("       2. 数据集名称不匹配（当前: %s）" % DATASET_NAME)
        print("       3. 文档语言（中文）导致实体提取失败")
        return

    print(f"共提取到 {len(all_triples)} 条三元组 (来自宽泛查询，不代表全量):")
    for i, item in enumerate(all_triples[:50], 1):
        print(fmt_result(i, item, "TRIPLET_COMPLETION"))
    if len(all_triples) > 50:
        print(f"\n... 还有 {len(all_triples) - 50} 条，使用具体关键词查询以获取更多")
    print("\n[提示] 用 'python cognee-search.py <关键词> --type TRIPLET_COMPLETION' 查询具体实体")


async def do_search(query: str, search_type: str):
    print(f"\n查询: {query!r}")
    print(f"类型: {search_type}  |  数据集: {DATASET_NAME}")
    if search_type == "GRAPH_COMPLETION":
        print("[注意] GRAPH_COMPLETION 在图谱上下文不足时会产生幻觉，建议先用 --type TRIPLET_COMPLETION 确认已索引的内容")
    print("─" * 56)

    from cognee.modules.search.types.SearchType import SearchType
    stype = getattr(SearchType, search_type, SearchType.TRIPLET_COMPLETION)

    try:
        results = await cognee.search(
            query,
            query_type=stype,
            datasets=DATASET_NAME,
        )
    except Exception as e:
        print(f"[ERROR] 搜索失败: {e}")
        if search_type == "GRAPH_COMPLETION":
            print("[TIP]  请改用 --type TRIPLET_COMPLETION 查看图谱中实际提取的内容")
        return

    if not results:
        print("(无结果 — 图谱中可能没有与此查询相关的节点)")
        if search_type in ("CHUNKS", "RAG_COMPLETION", "SUMMARIES"):
            print("     这些类型需要向量集合（DocumentChunk_text/TextSummary_text）")
            print("     建议改用: --type TRIPLET_COMPLETION")
        return

    print(f"共 {len(results)} 条结果:")
    for i, item in enumerate(results, 1):
        print(fmt_result(i, item, search_type))


async def interactive():
    print("=" * 56)
    print("  Cognee 文档搜索  (输入 'q' 退出)")
    print(f"  可用类型: {', '.join(SEARCH_TYPES)}")
    print("=" * 56)

    default_type = "SUMMARIES"
    while True:
        try:
            query = input(f"\n查询内容 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not query or query.lower() in ("q", "quit", "exit"):
            print("退出。")
            break

        type_input = input(f"搜索类型 [{default_type}] > ").strip()
        stype = type_input.upper() if type_input else default_type
        if stype not in SEARCH_TYPES:
            print(f"[WARN] 未知类型，使用 {default_type}")
            stype = default_type

        await do_search(query, stype)


async def main():
    parser = argparse.ArgumentParser(description="查询 Cognee 知识图谱")
    parser.add_argument("query", nargs="?", help="查询内容（省略则进入交互模式）")
    parser.add_argument(
        "--type", default="GRAPH_COMPLETION",
        choices=SEARCH_TYPES,
        help="搜索类型（默认 TRIPLET_COMPLETION，返回图谱中实际提取的实体关系）",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出图谱中已索引的所有实体/关系（诊断用）",
    )
    args = parser.parse_args()

    if not LLM_API_KEY:
        print("[ERROR] LLM_API_KEY 未设置，请在 .env 中配置。")
        return

    if args.list:
        await do_list()
    elif args.query:
        await do_search(args.query, args.type)
    else:
        await interactive()

# 可用搜索类型备忘：
# TRIPLET_COMPLETION  图谱三元组（不需向量，查真实抽取内容）
# GRAPH_COMPLETION    图谱 RAG（LLM 补全，可能幻觉）
# SUMMARIES           文档摘要（需 TextSummary_text 向量集合）
# CHUNKS              原始文本片段（需 DocumentChunk_text 向量集合）
# RAG_COMPLETION      纯向量 RAG（同上，需向量集合）


if __name__ == "__main__":
    asyncio.run(main())
