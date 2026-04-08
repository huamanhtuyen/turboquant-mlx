#!/usr/bin/env python3
"""TurboQuant-MLX — OpenAI-compatible API Server

Compatible với: Kilo Code, Cursor, Continue.dev, OpenClaw, LM Studio clients
Hỗ trợ: streaming SSE, tool calling (Qwen2.5/Hermes XML format), multi-turn

Usage:
    python server.py
    python server.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
    python server.py --model mlx-community/Hermes-3-Llama-3.1-8B-8bit --bits 8
    python server.py --port 8080 --host 0.0.0.0
"""

import asyncio
import json
import re
import time
import uuid
import argparse
import threading
import sys
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import mlx.core as mx
from mlx_lm import load
from turbo_cache import TurboQuantCache

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uvicorn


# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_model_id: str = ""
_bits: int = 8
_gen_lock = threading.Lock()   # MLX/Metal: serialize all inference calls
_MAX_PROMPT_TOKENS: int = 8192  # hard cap — override with --max-prompt-tokens
_req_counter = 0               # global request counter

# ANSI colors
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "cyan":   "\033[36m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "blue":   "\033[34m",
    "magenta":"\033[35m",
}


@dataclass
class _ReqStats:
    req_id:          str   = ""
    endpoint:        str   = ""
    prompt_tokens:   int   = 0
    prompt_preview:  str   = ""
    completion_tokens: int = 0
    prefill_ms:      float = 0.0
    decode_s:        float = 0.0
    cache_mb:        float = 0.0
    gpu_peak_gb:     float = 0.0
    ram_gb:          float = 0.0
    stream:          bool  = False

    @property
    def tok_per_s(self) -> float:
        if self.decode_s <= 0 or self.completion_tokens <= 1:
            return 0.0
        # decode_s covers tokens 2..N, so use completion_tokens-1
        return (self.completion_tokens - 1) / self.decode_s


def _collect_memory(stats: _ReqStats):
    """Populate GPU peak and RAM into stats object."""
    try:
        stats.gpu_peak_gb = mx.get_peak_memory() / 2**30
    except AttributeError:
        try:
            stats.gpu_peak_gb = mx.metal.get_peak_memory() / 2**30  # type: ignore
        except Exception:
            stats.gpu_peak_gb = 0.0
    if _HAS_PSUTIL:
        try:
            stats.ram_gb = _psutil.Process().memory_info().rss / 2**30
        except Exception:
            pass


def _cache_mb_now() -> float:
    """Total Metal GPU active memory in MB (model weights + KV cache + activations)."""
    try:
        return mx.get_active_memory() / 2**20
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / 2**20  # type: ignore
        except Exception:
            return 0.0


def _print_stats(stats: _ReqStats, counter: int):
    """Print a compact, color-coded stats block to console."""
    C = _C
    mode = "stream" if stats.stream else "full"
    tok_s = stats.tok_per_s
    total_s = (stats.prefill_ms / 1000) + stats.decode_s

    # Speed color
    if tok_s >= 20:
        spd_c = C["green"]
    elif tok_s >= 10:
        spd_c = C["yellow"]
    else:
        spd_c = C["red"]

    sep = C["dim"] + "─" * 64 + C["reset"]
    print(sep)
    print(
        f"{C['bold']}{C['cyan']}#{counter:04d}{C['reset']}  "
        f"{C['bold']}{stats.endpoint}{C['reset']}  "
        f"{C['dim']}({mode}){C['reset']}"
    )
    # Prompt info
    preview = stats.prompt_preview[:80].replace("\n", " ")
    print(
        f"  {C['dim']}Prompt{C['reset']}   "
        f"{C['yellow']}{stats.prompt_tokens:,} tok{C['reset']}  "
        f"{C['dim']}│ {preview!r}{C['reset']}"
    )
    # Timing
    print(
        f"  {C['dim']}Timing{C['reset']}   "
        f"prefill {C['blue']}{stats.prefill_ms:.0f}ms{C['reset']}  │  "
        f"decode  {spd_c}{tok_s:.1f} tok/s{C['reset']}  │  "
        f"{C['magenta']}{stats.completion_tokens} tokens{C['reset']}  │  "
        f"total {total_s:.1f}s"
    )
    # Memory
    gpu_str = f"{C['blue']}{stats.gpu_peak_gb:.2f} GB{C['reset']}" if stats.gpu_peak_gb > 0 else "n/a"
    ram_str = f"{stats.ram_gb:.2f} GB" if stats.ram_gb > 0 else ""
    # cache_mb = total Metal active memory (model ~4GB + actual KV cache)
    # Actual KV cache ≈ cache_mb - model_weights; shown as reference only
    active_str = f"{stats.cache_mb:.0f} MB" if stats.cache_mb > 0 else ""
    mem_parts = [f"GPU peak {gpu_str}"]
    if ram_str:
        mem_parts.append(f"RAM {ram_str}")
    if active_str:
        mem_parts.append(f"Metal active {active_str}")
    print(f"  {C['dim']}Memory{C['reset']}   " + "  │  ".join(mem_parts))
    print(sep)


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_model(model_id: str, bits: int):
    global _model, _tokenizer, _model_id, _bits
    print(f"\nLoading {model_id} ...")
    _model, _tokenizer = load(model_id)
    _model_id = model_id
    _bits = bits
    cache_label = "FP16" if bits == 0 else f"TurboQuant {bits}-bit"
    print(f"  ✓ {len(_model.layers)} layers | head_dim={_model.head_dim} "
          f"| kv_heads={_model.n_kv_heads} | cache={cache_label}")


def _make_cache():
    kv_heads = (
        [_model.n_kv_heads] * len(_model.layers)
        if isinstance(_model.n_kv_heads, int)
        else _model.n_kv_heads
    )
    return [TurboQuantCache(_model.head_dim, n, bits=_bits) for n in kv_heads]


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _format_prompt(messages: list, tools: Optional[list] = None) -> str:
    """Apply tokenizer chat template. Inject tools into system prompt if needed."""
    msgs = list(messages)

    # Inject tool definitions into system prompt (Qwen2.5/Hermes style)
    if tools:
        tool_json = json.dumps(tools, ensure_ascii=False, indent=2)
        tool_block = (
            "You are a helpful assistant. You have access to the following tools.\n"
            "Use them if required. Tool definitions are within <tools></tools> tags.\n\n"
            f"<tools>\n{tool_json}\n</tools>\n\n"
            "For each tool call, output a JSON object within <tool_call></tool_call> tags:\n"
            "<tool_call>\n"
            '{"name": "<tool_name>", "arguments": {<args>}}\n'
            "</tool_call>"
        )
        # Prepend or merge into existing system message
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": tool_block + "\n\n" + msgs[0]["content"]}
        else:
            msgs.insert(0, {"role": "system", "content": tool_block})

    if hasattr(_tokenizer, "apply_chat_template"):
        try:
            return _tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback: plain concatenation
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


# ── Core inference (synchronous, must run under _gen_lock) ────────────────────

def _iter_tokens(prompt: str, max_tokens: int, temp: float,
                  stats: Optional[_ReqStats] = None) -> Iterator[str]:
    """Yield decoded text fragments token by token.
    Caller is responsible for holding _gen_lock.
    Optionally populates a _ReqStats object with prefill/decode timing.
    """
    cache = _make_cache()
    input_ids = mx.array(_tokenizer.encode(prompt))[None]

    if stats is not None:
        # Use new API (mx.reset_peak_memory), fall back to deprecated for older MLX
        try:
            mx.reset_peak_memory()
        except AttributeError:
            mx.metal.reset_peak_memory()  # type: ignore[attr-defined]

    # Prefill
    t_prefill = time.perf_counter()
    logits = _model(input_ids, cache=cache)
    mx.eval(logits)
    prefill_ms = (time.perf_counter() - t_prefill) * 1000

    if stats is not None:
        stats.prefill_ms = prefill_ms

    # Sample first token
    token = (
        mx.argmax(logits[:, -1, :], axis=-1)
        if temp == 0.0
        else mx.random.categorical(logits[:, -1, :] / temp)
    )

    eos_id = getattr(_tokenizer, "eos_token_id", None)
    n_decoded = 0
    t_decode = time.perf_counter()

    for _ in range(max_tokens):
        tid = token.item()
        if eos_id is not None and tid == eos_id:
            break

        yield _tokenizer.decode([tid])
        n_decoded += 1

        logits = _model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        token = (
            mx.argmax(logits[:, -1, :], axis=-1)
            if temp == 0.0
            else mx.random.categorical(logits[:, -1, :] / temp)
        )

    if stats is not None:
        stats.decode_s         = time.perf_counter() - t_decode
        stats.completion_tokens = n_decoded
        stats.cache_mb          = _cache_mb_now()
        _collect_memory(stats)


def _run_full(prompt: str, max_tokens: int, temp: float,
              stats: Optional[_ReqStats] = None) -> str:
    """Generate full response synchronously (non-streaming)."""
    with _gen_lock:
        parts = list(_iter_tokens(prompt, max_tokens, temp, stats=stats))
    return "".join(parts)


# ── Tool call parsing ─────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Partial open-tag prefixes to detect in-progress tags while streaming
_PARTIAL_OPEN = ["<", "<t", "<to", "<too", "<tool", "<tool_", "<tool_c",
                 "<tool_ca", "<tool_cal", "<tool_call", "<tool_call>"]


def _parse_tool_calls(text: str) -> tuple[list, str]:
    """Extract <tool_call>...</tool_call> blocks, return (tool_calls, remaining_text)."""
    tool_calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                },
            })
        except (json.JSONDecodeError, KeyError):
            pass
    remaining = _TOOL_CALL_RE.sub("", text).strip()
    return tool_calls, remaining


def _ends_with_partial_tag(text: str) -> bool:
    for prefix in _PARTIAL_OPEN:
        if text.endswith(prefix):
            return True
    return False


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk(req_id: str, created: int, delta: dict,
           finish_reason: Optional[str] = None) -> str:
    return _sse({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })


# ── Streaming generator ───────────────────────────────────────────────────────

async def _stream_response(prompt: str, max_tokens: int, temp: float,
                            req_id: str, created: int,
                            stats: Optional[_ReqStats] = None):
    """Async SSE generator. Runs MLX inference in a background thread."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            with _gen_lock:
                for text in _iter_tokens(prompt, max_tokens, temp, stats=stats):
                    loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # First chunk: role
    yield _chunk(req_id, created, {"role": "assistant", "content": ""})

    pending = ""       # chars buffered waiting for safe-to-send decision
    tool_buf = ""      # chars accumulated inside a <tool_call>…</tool_call>
    in_tool = False

    while True:
        item = await queue.get()

        if isinstance(item, Exception):
            raise item
        if item is None:
            break

        if not in_tool:
            pending += item

            if "<tool_call>" in pending:
                # Flush text before the tag, then enter tool-call mode
                tag_pos = pending.index("<tool_call>")
                before = pending[:tag_pos]
                if before:
                    yield _chunk(req_id, created, {"content": before})
                tool_buf = pending[tag_pos:]
                pending = ""
                in_tool = True

            elif _ends_with_partial_tag(pending):
                # Might be about to open a tool tag — hold in pending
                pass

            else:
                # Safe to stream immediately
                yield _chunk(req_id, created, {"content": pending})
                pending = ""

        else:
            tool_buf += item
            if "</tool_call>" in tool_buf:
                tool_calls, leftover = _parse_tool_calls(tool_buf)
                if tool_calls:
                    yield _chunk(req_id, created,
                                 {"content": None, "tool_calls": tool_calls},
                                 finish_reason="tool_calls")
                in_tool = False
                tool_buf = ""
                pending = leftover  # text after the closing tag

    # Flush any remaining pending text
    if pending:
        yield _chunk(req_id, created, {"content": pending})

    thread.join()

    yield _chunk(req_id, created, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── FastAPI app ───────────────────────────────────────────────────────────────

class _PrefixFixMiddleware(BaseHTTPMiddleware):
    """Rewrite double-prefix paths like /v1/v1/models → /v1/models.
    Some clients (Kilo Code, etc.) append /v1 to a Base URL that already
    contains /v1, producing invalid doubled paths.
    """
    async def dispatch(self, request: Request, call_next):
        path = request.scope["path"]
        # Strip one leading /v1 if the path starts with /v1/v1/
        if path.startswith("/v1/v1/"):
            request.scope["path"] = path[3:]  # e.g. /v1/v1/models → /v1/models
        return await call_next(request)


app = FastAPI(
    title="TurboQuant-MLX API",
    description="OpenAI-compatible local API with KV cache compression",
    version="1.0.0",
)

# Must be added BEFORE CORSMiddleware so it runs first
app.add_middleware(_PrefixFixMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _model_id,
        "cache_bits": _bits,
        "cache_type": "FP16" if _bits == 0 else f"TurboQuant-{_bits}bit",
    }


# ── OpenAI /v1/models ─────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "turboquant-mlx",
            "permission": [],
        }],
    }

@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    return {
        "id": _model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "turboquant-mlx",
    }


# ── OpenAI /v1/chat/completions ───────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _req_counter
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages: list = body.get("messages", [])
    tools: Optional[list] = body.get("tools") or body.get("functions")
    max_tokens: int = body.get("max_tokens") or body.get("max_new_tokens") or 1024
    temperature: float = float(body.get("temperature", 0.0))
    stream: bool = body.get("stream", False)

    if not messages:
        raise HTTPException(400, "messages is required")

    prompt = _format_prompt(messages, tools=tools)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    _req_counter += 1
    counter = _req_counter

    # Build stats object — prompt info populated here, rest filled during inference
    prompt_tokens = len(_tokenizer.encode(prompt))
    # Extract a readable preview from the last user message
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        prompt,
    )
    stats = _ReqStats(
        req_id=req_id,
        endpoint="/v1/chat/completions",
        prompt_tokens=prompt_tokens,
        prompt_preview=str(last_user)[:200],
        stream=stream,
    )

    # ── Hard cap on prompt size ───────────────────────────────────────
    if prompt_tokens > _MAX_PROMPT_TOKENS:
        C = _C
        original_tokens = prompt_tokens
        # Truncate token ids then decode back to string
        token_ids = _tokenizer.encode(prompt)
        # Keep last _MAX_PROMPT_TOKENS tokens (tail = most recent context)
        kept_ids = token_ids[-_MAX_PROMPT_TOKENS:]
        prompt = _tokenizer.decode(kept_ids)
        prompt_tokens = len(kept_ids)
        stats.prompt_tokens = prompt_tokens
        print(
            f"\n{C['bold']}{C['red']}⚠ PROMPT TRUNCATED{C['reset']}  "
            f"{original_tokens:,} → {prompt_tokens:,} tokens  "
            f"{C['dim']}(limit: {_MAX_PROMPT_TOKENS:,} | use --max-prompt-tokens N to change){C['reset']}"
        )

    if stream:
        async def _streaming_with_stats():
            async for chunk in _stream_response(
                prompt, max_tokens, temperature, req_id, created, stats=stats
            ):
                yield chunk
            # Print stats after stream completes
            _print_stats(stats, counter)

        return StreamingResponse(
            _streaming_with_stats(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    loop = asyncio.get_event_loop()
    full_text = await loop.run_in_executor(
        None, _run_full, prompt, max_tokens, temperature, stats
    )
    _print_stats(stats, counter)

    tool_calls, content = _parse_tool_calls(full_text)
    finish_reason = "tool_calls" if tool_calls else "stop"

    message: dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": stats.completion_tokens or len(_tokenizer.encode(full_text)),
            "total_tokens": prompt_tokens + (stats.completion_tokens or len(_tokenizer.encode(full_text))),
        },
    }


# ── OpenAI /v1/completions (raw text) ────────────────────────────────────────

@app.post("/v1/completions")
async def completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt: str = body.get("prompt", "")
    max_tokens: int = body.get("max_tokens") or 512
    temperature: float = float(body.get("temperature", 0.0))
    stream: bool = body.get("stream", False)

    if not prompt:
        raise HTTPException(400, "prompt is required")

    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:
        async def _text_stream():
            yield _sse({
                "id": req_id, "object": "text_completion.chunk",
                "created": created, "model": _model_id,
                "choices": [{"index": 0, "text": "", "finish_reason": None}],
            })
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _worker():
                try:
                    with _gen_lock:
                        for tok in _iter_tokens(prompt, max_tokens, temperature):
                            loop.call_soon_threadsafe(queue.put_nowait, tok)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=_worker, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse({
                    "id": req_id, "object": "text_completion.chunk",
                    "created": created, "model": _model_id,
                    "choices": [{"index": 0, "text": item, "finish_reason": None}],
                })
            yield _sse({
                "id": req_id, "object": "text_completion.chunk",
                "created": created, "model": _model_id,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            })
            yield "data: [DONE]\n\n"

        return StreamingResponse(_text_stream(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _run_full, prompt, max_tokens, temperature)
    return {
        "id": req_id, "object": "text_completion",
        "created": created, "model": _model_id,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(_tokenizer.encode(prompt)),
            "completion_tokens": len(_tokenizer.encode(text)),
            "total_tokens": len(_tokenizer.encode(prompt)) + len(_tokenizer.encode(text)),
        },
    }


# ── Ollama-compatible endpoints ───────────────────────────────────────────────

@app.get("/api/tags")
async def ollama_tags():
    short_name = _model_id.split("/")[-1]
    return {
        "models": [{
            "name": short_name,
            "model": short_name,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size": 0,
            "digest": "turboquant-mlx",
            "details": {
                "parent_model": _model_id,
                "format": "mlx",
                "family": "qwen",
                "parameter_size": "7B",
                "quantization_level": f"{_bits}bit" if _bits > 0 else "fp16",
            },
        }]
    }


@app.post("/api/show")
async def ollama_show(request: Request):
    """Ollama POST /api/show — return model metadata."""
    short_name = _model_id.split("/")[-1]
    return {
        "model": short_name,
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "template": "",
        "details": {
            "parent_model": _model_id,
            "format": "mlx",
            "family": "qwen",
            "families": ["qwen"],
            "parameter_size": "7B",
            "quantization_level": f"{_bits}bit" if _bits > 0 else "fp16",
        },
        "model_info": {
            "general.architecture": "qwen2",
            "general.name": short_name,
        },
    }


@app.post("/api/show/{model_name:path}")
async def ollama_show_named(model_name: str):
    """Ollama GET /api/show/<name> variant."""
    return await ollama_show(None)


@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Ollama /api/chat → forward to OpenAI handler."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages = body.get("messages", [])
    stream = body.get("stream", True)
    opts = body.get("options", {})

    # Translate to OpenAI format and reuse the same logic
    oai_body = {
        "messages": messages,
        "max_tokens": opts.get("num_predict", 1024),
        "temperature": opts.get("temperature", 0.0),
        "stream": stream,
    }

    class _MockRequest:
        async def json(self):
            return oai_body

    if stream:
        # Ollama streaming format: newline-delimited JSON (not SSE)
        prompt = _format_prompt(messages)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def _ollama_stream():
            collected = []
            async for sse_line in _stream_response(
                prompt, oai_body["max_tokens"], oai_body["temperature"], req_id, created
            ):
                if sse_line.startswith("data: ") and not sse_line.startswith("data: [DONE]"):
                    try:
                        chunk = json.loads(sse_line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        content_piece = delta.get("content") or ""
                        collected.append(content_piece)
                        yield json.dumps({
                            "model": _model_id,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "message": {"role": "assistant", "content": content_piece},
                            "done": False,
                        }, ensure_ascii=False) + "\n"
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
            # Final done message
            yield json.dumps({
                "model": _model_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }, ensure_ascii=False) + "\n"

        return StreamingResponse(_ollama_stream(), media_type="application/x-ndjson")

    # Non-streaming Ollama
    resp = await chat_completions(_MockRequest())
    content = resp["choices"][0]["message"].get("content", "")
    return {
        "model": _model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "eval_count": resp.get("usage", {}).get("completion_tokens", 0),
    }


@app.post("/api/generate")
async def ollama_generate(request: Request):
    """Ollama /api/generate (raw prompt)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    opts = body.get("options", {})
    max_tokens = opts.get("num_predict", 512)
    temperature = opts.get("temperature", 0.0)

    if not prompt:
        raise HTTPException(400, "prompt is required")

    if stream:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _worker():
            try:
                with _gen_lock:
                    for tok in _iter_tokens(prompt, max_tokens, temperature):
                        loop.call_soon_threadsafe(queue.put_nowait, tok)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        async def _gen_stream():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps({
                    "model": _model_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response": item,
                    "done": False,
                }, ensure_ascii=False) + "\n"
            yield json.dumps({
                "model": _model_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": "",
                "done": True,
                "done_reason": "stop",
            }, ensure_ascii=False) + "\n"

        return StreamingResponse(_gen_stream(), media_type="application/x-ndjson")

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _run_full, prompt, max_tokens, temperature)
    return {
        "model": _model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": text,
        "done": True,
    }


# ── LM Studio WebSocket endpoints ───────────────────────────────────────────────
# Kilo Code in "LM Studio" mode connects to /system and /llm via WebSocket.
# /system  — heartbeat / hardware info channel
# /llm     — streaming inference channel (JSON-RPC style, LM Studio SDK protocol)

@app.websocket("/system")
async def ws_system(ws: WebSocket):
    """LM Studio /system WebSocket — hardware/status heartbeat."""
    await ws.accept()
    try:
        # Send initial system greeting expected by LM Studio SDK
        greeting = {
            "type": "serverStatus",
            "status": "ready",
            "model": _model_id,
            "gpuInfo": [{"name": "Apple M-series", "totalMemory": 16384}],
            "cpuInfo": {"numPhysicalCores": 8, "numLogicalCores": 8},
        }
        await ws.send_json(greeting)

        # Keep-alive loop: respond to any pings
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
                msg_type = msg.get("type", "")
                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                elif msg_type == "keepAlive":
                    await ws.send_json({"type": "keepAliveAck"})
                else:
                    # Echo unknown messages as ack
                    await ws.send_json({"type": "ack", "echo": msg_type})
            except asyncio.TimeoutError:
                # Send heartbeat
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/llm")
async def ws_llm(ws: WebSocket):
    """LM Studio /llm WebSocket — streaming inference channel.

    LM Studio SDK protocol (simplified):
      client → server:  {type: "predict", extra: {requestId}, params: {messages, ...}}
      server → client:  {type: "fragment", requestId, fragment: {content: "..."}}
                         {type: "success", requestId}  (when done)
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                msg = await ws.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                break

            msg_type = msg.get("type", "")

            if msg_type in ("predict", "chat", "completions"):
                extra = msg.get("extra", {})
                params = msg.get("params", msg)  # some versions inline params
                request_id = extra.get("requestId", str(uuid.uuid4()))
                messages = params.get("messages", [])
                max_tokens = params.get("max_tokens") or params.get("maxPredictedTokens") or 1024
                temperature = float(params.get("temperature", 0.0))

                if not messages:
                    await ws.send_json({"type": "error", "requestId": request_id,
                                        "error": "messages is required"})
                    continue

                prompt = _format_prompt(messages)
                queue: asyncio.Queue = asyncio.Queue()

                def _worker():
                    try:
                        with _gen_lock:
                            for tok in _iter_tokens(prompt, max_tokens, temperature):
                                loop.call_soon_threadsafe(queue.put_nowait, tok)
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, exc)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                threading.Thread(target=_worker, daemon=True).start()

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        await ws.send_json({
                            "type": "error", "requestId": request_id,
                            "error": str(item),
                        })
                        break
                    # Stream fragment
                    await ws.send_json({
                        "type": "fragment",
                        "requestId": request_id,
                        "fragment": {"content": item},
                    })

                # Done
                await ws.send_json({
                    "type": "success",
                    "requestId": request_id,
                    "stats": {"stop_reason": "end"},
                })

            elif msg_type == "keepAlive":
                await ws.send_json({"type": "keepAliveAck"})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

            else:
                # Unknown message — ack gracefully
                await ws.send_json({"type": "ack", "receivedType": msg_type})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant-MLX — OpenAI-compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Kilo Code / Cursor (coding)
  python server.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

  # OpenClaw / Hermes agent
  python server.py --model mlx-community/Hermes-3-Llama-3.1-8B-8bit --bits 8

  # FP16 baseline (no compression)
  python server.py --bits 0

  # Limit prompt size (prevent Kilo Code / Cursor from sending 15k token prompts)
  python server.py --max-prompt-tokens 6000

  # Expose to local network
  python server.py --host 0.0.0.0 --port 8080
        """,
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="HuggingFace model repo (default: Qwen2.5-7B-Instruct-4bit)",
    )
    parser.add_argument(
        "--bits", type=int, default=8, choices=[0, 4, 8],
        help="KV cache quantization: 0=FP16, 8=TurboQuant-8bit (default: 8)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=8192,
        help="Hard cap on prompt length in tokens. Prompts exceeding this are "
             "truncated from the front (keeping most recent context). "
             "Set lower (e.g. 4096) if Kilo Code sends huge codebase context. (default: 8192)",
    )
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    global _MAX_PROMPT_TOKENS
    _load_model(args.model, args.bits)
    _MAX_PROMPT_TOKENS = args.max_prompt_tokens

    short = args.model.split("/")[-1]
    cache_label = "FP16" if args.bits == 0 else f"TurboQuant {args.bits}-bit"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          TurboQuant-MLX API Server  🚀                   ║
╠══════════════════════════════════════════════════════════╣
║  Model     : {args.model:<43} ║
║  Cache     : {cache_label:<43} ║
║  Prompt cap: {args.max_prompt_tokens:<,} tokens{'':<30} ║
║  URL       : http://{args.host}:{args.port:<36} ║
╠══════════════════════════════════════════════════════════╣
║  Kilo Code / Cursor / Continue.dev config:               ║
║    Provider : OpenAI Compatible                          ║
║    Base URL : http://{args.host}:{args.port}/v1{'':<28} ║
║    API Key  : not-needed                                 ║
║    Model ID : {short:<44} ║
╠══════════════════════════════════════════════════════════╣
║  Ollama-compatible (OpenClaw):                           ║
║    Base URL : http://{args.host}:{args.port}{'':<35} ║
╚══════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
