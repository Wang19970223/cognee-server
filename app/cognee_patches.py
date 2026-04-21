"""
app/cognee_patches.py
─────────────────────────────────────────────────────────────────────────────
Patches that MUST be applied before any `import cognee` call:
  1. litellm.aembedding  → bypass litellm, call DashScope /embeddings directly
                           (DashScope max 10 items/batch; batched in parallel)
  2. litellm.acompletion → fix qwen-max function-call arguments with bare
                           newlines that break JSON parsing
  3. tiktoken alias      → map DashScope embedding model names to cl100k_base

Call `apply_patches(settings)` once during application startup.
"""

from __future__ import annotations

import json as _json


def apply_patches(llm_api_key: str, llm_endpoint: str, http_proxy: str = "") -> None:
    """Apply all litellm + tiktoken patches. Must be called before `import cognee`."""
    import litellm as _litellm
    import httpx as _httpx
    import asyncio as _asyncio
    import tiktoken as _tiktoken

    # ── 1. DashScope embedding bypass ─────────────────────────────────────────
    async def _safe_aembedding(*args, **kwargs):
        model = kwargs.get("model", "").replace("openai/", "")
        api_base = kwargs.get("api_base") or llm_endpoint
        api_key = kwargs.get("api_key") or llm_api_key
        input_data = kwargs.get("input", [])
        if isinstance(input_data, str):
            input_data = [input_data]

        if not input_data:
            # Empty input — DashScope returns 422 for empty arrays.
            class _EmptyResp:
                data = []
            return _EmptyResp()

        client_kwargs: dict = {"timeout": 60, "verify": False}
        if http_proxy:
            client_kwargs["proxy"] = http_proxy

        BATCH_SIZE = 10
        batches = [input_data[i: i + BATCH_SIZE] for i in range(0, len(input_data), BATCH_SIZE)]

        async def _fetch_batch(batch):
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

        batch_results = await _asyncio.gather(*[_fetch_batch(b) for b in batches])

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

    # ── 2. qwen/DashScope acompletion fix ────────────────────────────────────
    # Fixes two issues:
    # a) qwen-max/-turbo function-call arguments may contain bare newlines
    #    that break strict JSON parsing → round-trip through json.loads(strict=False)
    # b) DashScope OpenAI-compatible endpoint rejects `parallel_tool_calls` and
    #    other OpenAI-only parameters with HTTP 422 Unprocessable Entity
    _orig_acompletion = _litellm.acompletion

    # Parameters that DashScope does not accept in its OpenAI-compatible API
    _DASHSCOPE_UNSUPPORTED_PARAMS = frozenset({
        "parallel_tool_calls",
        "stream_options",
        "logprobs",
        "top_logprobs",
        "service_tier",
    })

    def _fix_tool_args(arguments: str) -> str:
        try:
            return _json.dumps(_json.loads(arguments, strict=False), ensure_ascii=False)
        except Exception:
            return arguments

    async def _safe_acompletion(*args, **kwargs):
        # Strip unsupported params before sending to DashScope
        for _p in _DASHSCOPE_UNSUPPORTED_PARAMS:
            kwargs.pop(_p, None)
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

    # ── 3. tiktoken alias for DashScope embedding model names ─────────────────
    _orig_enc = _tiktoken.encoding_for_model

    _ALIAS = {
        "text-embedding-v2": "cl100k_base",
        "text-embedding-v3": "cl100k_base",
    }

    def _patched_enc(model_name):
        return (
            _tiktoken.get_encoding(_ALIAS[model_name])
            if model_name in _ALIAS
            else _orig_enc(model_name)
        )

    _tiktoken.encoding_for_model = _patched_enc

    # ── 4. tiktoken BPE download: use proxy + skip SSL verify ─────────────────
    # On corporate networks the self-signed proxy cert causes SSL errors when
    # tiktoken tries to fetch cl100k_base.tiktoken from openaipublic.blob.core.windows.net
    import tiktoken.load as _tiktoken_load
    import requests as _requests
    import urllib3 as _urllib3
    _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)

    _orig_read_file = _tiktoken_load.read_file

    def _patched_read_file(blobpath: str) -> bytes:
        proxies = {"http": http_proxy, "https": http_proxy} if http_proxy else None
        try:
            resp = _requests.get(blobpath, proxies=proxies, verify=False, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception:
            # Fall back to original if proxy download also fails
            return _orig_read_file(blobpath)

    _tiktoken_load.read_file = _patched_read_file
