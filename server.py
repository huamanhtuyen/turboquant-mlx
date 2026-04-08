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
from typing import Iterator, Optional

import mlx.core as mx
from mlx_lm import load
from turbo_cache import TurboQuantCache

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn


# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_model_id: str = ""
_bits: int = 8
_gen_lock = threading.Lock()   # MLX/Metal: serialize all inference calls


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

def _iter_tokens(prompt: str, max_tokens: int, temp: float) -> Iterator[str]:
    """Yield decoded text fragments token by token.
    Caller is responsible for holding _gen_lock.
    """
    cache = _make_cache()
    input_ids = mx.array(_tokenizer.encode(prompt))[None]

    # Prefill
    logits = _model(input_ids, cache=cache)
    mx.eval(logits)

    # Sample
    token = (
        mx.argmax(logits[:, -1, :], axis=-1)
        if temp == 0.0
        else mx.random.categorical(logits[:, -1, :] / temp)
    )

    eos_id = getattr(_tokenizer, "eos_token_id", None)

    for _ in range(max_tokens):
        tid = token.item()
        if eos_id is not None and tid == eos_id:
            break

        yield _tokenizer.decode([tid])

        logits = _model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        token = (
            mx.argmax(logits[:, -1, :], axis=-1)
            if temp == 0.0
            else mx.random.categorical(logits[:, -1, :] / temp)
        )


def _run_full(prompt: str, max_tokens: int, temp: float) -> str:
    """Generate full response synchronously (non-streaming)."""
    with _gen_lock:
        parts = list(_iter_tokens(prompt, max_tokens, temp))
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
                            req_id: str, created: int):
    """Async SSE generator. Runs MLX inference in a background thread."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            with _gen_lock:
                for text in _iter_tokens(prompt, max_tokens, temp):
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

app = FastAPI(
    title="TurboQuant-MLX API",
    description="OpenAI-compatible local API with KV cache compression",
    version="1.0.0",
)

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

    if stream:
        return StreamingResponse(
            _stream_response(prompt, max_tokens, temperature, req_id, created),
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
        None, _run_full, prompt, max_tokens, temperature
    )

    tool_calls, content = _parse_tool_calls(full_text)
    finish_reason = "tool_calls" if tool_calls else "stop"

    message: dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None

    prompt_tokens = len(_tokenizer.encode(prompt))
    completion_tokens = len(_tokenizer.encode(full_text))

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
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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

  # Expose to local network
  python server.py --host 0.0.0.0 --port 8080
        """,
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        help="HuggingFace model repo (default: Qwen2.5-Coder-7B-Instruct-4bit)",
    )
    parser.add_argument(
        "--bits", type=int, default=8, choices=[0, 4, 8],
        help="KV cache quantization: 0=FP16, 8=TurboQuant-8bit (default: 8)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    _load_model(args.model, args.bits)

    short = args.model.split("/")[-1]
    cache_label = "FP16" if args.bits == 0 else f"TurboQuant {args.bits}-bit"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          TurboQuant-MLX API Server  🚀                   ║
╠══════════════════════════════════════════════════════════╣
║  Model  : {args.model:<46} ║
║  Cache  : {cache_label:<46} ║
║  URL    : http://{args.host}:{args.port:<39} ║
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
