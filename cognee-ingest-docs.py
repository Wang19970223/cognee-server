"""
cognee-ingest-docs.py
────────────────────────────────────────────────────────────────
将 docs/design 目录下的所有 Markdown 文档批量导入 Cognee 知识图谱。
配置全部从 .env 文件读取，无需修改本脚本。

快速开始：
  1. 复制 .env.example → .env，填入真实 API Key
  2. python cognee-ingest-docs.py

可选：在 .env 中设置 COGNEE_PRUNE=true 以在导入前清空旧数据
────────────────────────────────────────────────────────────────
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Windows 终端/管道下强制 UTF-8 输出，避免 GBK 编码报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 1. 加载 .env（必须在 import cognee 之前完成所有 os.environ 设置）
load_dotenv()

# ---- 从 .env 读取，提供合理默认值 ----
PROXY          = os.environ.get("HTTP_PROXY", "")
LLM_API_KEY    = os.environ.get("LLM_API_KEY", "")
LLM_ENDPOINT   = os.environ.get("LLM_ENDPOINT",
                     "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "openai/qwen-max")
EMBED_MODEL    = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-v3")
EMBED_DIMS     = os.environ.get("EMBEDDING_DIMENSIONS", "1024")
DATASET_NAME   = os.environ.get("COGNEE_DATASET", "generic-graphic-ui-docs")
DOCS_DIR       = os.environ.get("DOCS_DIR",        "docs/design")
DO_PRUNE       = os.environ.get("COGNEE_PRUNE",    "false").lower() == "true"

# ---- 注入供 cognee / litellm 使用的环境变量 ----
os.environ["LLM_PROVIDER"]          = "openai"
os.environ["LLM_MODEL"]             = LLM_MODEL
os.environ["LLM_ENDPOINT"]          = LLM_ENDPOINT
os.environ["EMBEDDING_PROVIDER"]    = "openai"
os.environ["EMBEDDING_MODEL"]       = EMBED_MODEL
os.environ["EMBEDDING_ENDPOINT"]    = LLM_ENDPOINT
os.environ["EMBEDDING_API_KEY"]     = LLM_API_KEY
os.environ.pop("EMBEDDING_DIMENSIONS", None)
os.environ["EMBEDDING_DIMENSIONS"]  = EMBED_DIMS
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["TELEMETRY_DISABLED"]    = "1"
os.environ["LOG_LEVEL"]             = "ERROR"
os.environ["COGNEE_LOG_LEVEL"]      = "ERROR"
# 关闭多用户访问控制，否则 add/cognify/search 可能跑在不同用户上下文
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

if PROXY:
    os.environ["HTTP_PROXY"]  = PROXY
    os.environ["HTTPS_PROXY"] = PROXY

# ── 2. Patch litellm.aembedding
#    绕过 litellm，直接调 DashScope，避免 UnsupportedParamsError
import litellm as _litellm
import httpx as _httpx
import asyncio as _asyncio


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
#    qwen-max function call arguments 里可能含未转义换行符，
#    用 json.loads(strict=False)+json.dumps round-trip 修复。
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

# ── 3. tiktoken 别名（DashScope 模型名不在 tiktoken 内置列表）
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


# ════════════════════════════════════════════════════════════
# 文档收集
# ════════════════════════════════════════════════════════════

def collect_docs(docs_dir: str) -> list[tuple[str, str, Path]]:
    """
    递归收集 docs_dir 下所有 .md 文件。
    返回 [(relative_path, content, abs_path), ...]，按文件名排序。
    """
    base = Path(docs_dir)
    if not base.is_absolute():
        base = Path.cwd() / base

    if not base.exists():
        raise FileNotFoundError(f"文档目录不存在: {base}")

    files = sorted(base.rglob("*.md"))
    if not files:
        raise RuntimeError(f"在 {base} 下未找到任何 .md 文件")

    docs = []
    for f in files:
        try:
            rel = f.relative_to(Path.cwd())
            content = f.read_text(encoding="utf-8")
            docs.append((str(rel), content, f))
            size_kb = len(content.encode()) / 1024
            print(f"  [OK]   {rel}  ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")
    return docs


# ════════════════════════════════════════════════════════════
# 导入流程
# ════════════════════════════════════════════════════════════

async def ingest(docs: list[tuple[str, str, Path]], dataset: str):
    # 构造文本：在每篇文档最前面加上文件名作为标题，便于知识图谱追踪来源
    texts = [
        f"<!-- source: {rel} -->\n# {Path(rel).stem}\n\n{content}"
        for rel, content, _ in docs
    ]

    print(f"\n[1/4] cognee.add  ── 添加 {len(texts)} 篇文档到数据集 '{dataset}' ...")
    await cognee.add(texts, dataset)
    print("       ✔ add 完成")

    print("\n[2/4] cognee.cognify  ── 构建知识图谱（耗时较长，请耐心等待）...")
    await cognee.cognify()
    print("       ✔ cognify 完成")

    
    print("\n[3/4] 正在生成三元组向量索引 (create_triplet_embeddings)...")
    try:
        from cognee.modules.users.methods import get_default_user
        from cognee.memify_pipelines.create_triplet_embeddings import create_triplet_embeddings
        user = await get_default_user()
        await create_triplet_embeddings(user=user, dataset=dataset)
        print("       ✔ create_triplet_embeddings 完成")
    except Exception as e:
        print(f"⚠️ create_triplet_embeddings 出错: {e}")


    print("\n[4/4] 验证搜索  ── 查询图谱是否有抽取到节点 ...");
    from cognee.modules.search.types.SearchType import SearchType as ST
    # INSIGHTS 直接返回图谱中提取的实体/关系三元组，无需 LLM 补全，最可靠
    # GRAPH_COMPLETION 在上下文不足时会产生幻觉，不适合用来做验证
    search_types = [ST.TRIPLET_COMPLETION, ST.GRAPH_COMPLETION, ST.SUMMARIES]
    results = None
    used_type = None
    for st in search_types:
        try:
            results = await cognee.search(
                "element stage graph design",
                query_type=st,
                datasets=dataset,
            )
            used_type = st
            break
        except Exception as _se:
            print(f"       ⚠ {st.value} 搜索不可用（{type(_se).__name__}），尝试下一个...")
    if results is None:
        print("       ⚠ 所有搜索类型均不可用，跳过验证（数据已写入，可稍后查询）")
    else:
        print(f"       ✔ 使用 {used_type.value} 返回 {len(results)} 条结果")
        if results:
            preview = str(results[0])[:300].replace("\n", " ")
            print(f"       首条: {preview}")
            print(f"\n  [诊断] 运行 python cognee-search.py --list 可查看所有已索引的节点")


# ════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════

async def main():
    print("=" * 64)
    print("  Cognee 文档导入脚本")
    print(f"  文档目录  : {DOCS_DIR}")
    print(f"  数据集    : {DATASET_NAME}")
    print(f"  LLM 模型  : {LLM_MODEL}")
    print(f"  Embedding : {EMBED_MODEL}  (dims={EMBED_DIMS})")
    print(f"  代理      : {PROXY or '（无）'}")
    print(f"  导入前清空: {'是' if DO_PRUNE else '否'}")
    print("=" * 64)

    if not LLM_API_KEY:
        print("\n[ERROR] LLM_API_KEY 未设置，请在 .env 中配置后重新运行。")
        return

    print("\n收集文档...")
    try:
        docs = collect_docs(DOCS_DIR)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[ERROR] {e}")
        return

    total_chars = sum(len(c) for _, c, _ in docs)
    print(f"\n共 {len(docs)} 篇（约 {total_chars // 1000} K 字符），开始导入...\n")

    if DO_PRUNE:
        print("[PRUNE] 清空旧数据...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        print("        ✔ prune 完成\n")

    try:
        await ingest(docs, DATASET_NAME)
        print("\n" + "=" * 64)
        print("  ✅  全部完成！")
        print(f"  数据集 '{DATASET_NAME}' 已就绪，可通过 cognee.search() 查询。")
        print("=" * 64)
    except Exception:
        print("\n[FAILED] 导入出错：")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
