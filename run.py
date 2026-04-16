#!/usr/bin/env python3
"""TurboQuant-MLX: KV cache compression demo.

Usage:
    python run.py                          # interactive chat
    python run.py --benchmark              # run benchmark
    python run.py --bits 3                 # use 3-bit quantization
    python run.py --model <hf-repo>        # use different model
"""

import sys
import time
import argparse
import mlx.core as mx
from mlx_lm import load
from turbo_cache import TurboQuantCache

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def make_turbo_cache(model, bits=4):
    """Create TurboQuantCache list matching model architecture."""
    kv_heads = (
        [model.n_kv_heads] * len(model.layers)
        if isinstance(model.n_kv_heads, int)
        else model.n_kv_heads
    )
    return [TurboQuantCache(model.head_dim, n, bits=bits) for n in kv_heads]


def _format_messages(tokenizer, messages):
    """Apply chat template to a list of messages, returning a formatted string."""
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for tokenizers without chat template
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def generate(model, tokenizer, prompt, cache_list, max_tokens=200, temp=0.0, stream=False):
    """Generate text with custom KV cache (single-turn, stateless)."""
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(formatted))[None]
    else:
        input_ids = mx.array(tokenizer.encode(prompt))[None]

    n_prompt = input_ids.shape[1]

    # Prefill
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache_list)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0

    # First token
    if temp == 0:
        token = mx.argmax(logits[:, -1, :], axis=-1)
    else:
        token = mx.random.categorical(logits[:, -1, :] / temp)

    tokens = [token.item()]
    if stream:
        sys.stdout.write(tokenizer.decode(tokens))
        sys.stdout.flush()

    # Decode loop
    t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache_list)
        mx.eval(logits)

        if temp == 0:
            token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            token = mx.random.categorical(logits[:, -1, :] / temp)

        tid = token.item()
        tokens.append(tid)

        if stream:
            sys.stdout.write(tokenizer.decode([tid]))
            sys.stdout.flush()

        if tid == tokenizer.eos_token_id:
            break

    decode_time = time.perf_counter() - t0
    n_decode = len(tokens)

    return {
        "text": tokenizer.decode(tokens),
        "tokens": n_decode,
        "prompt_tokens": n_prompt,
        "prefill_ms": prefill_time * 1000,
        "decode_tok_s": n_decode / decode_time if decode_time > 0 else 0,
    }


def generate_incremental(model, tokenizer, new_ids, cache_list,
                         max_tokens=200, temp=0.0, stream=False):
    """Prefill only new_ids into an existing cache, then decode response.

    Used for stateful multi-turn chat: cache accumulates KV state across turns
    so each call only processes the tokens added since the last turn.

    Returns:
        dict with keys: text, n_decode, n_prompt, prefill_ms, decode_tok_s
        n_cache_added: total tokens added to the cache this call
                       (= n_prompt + n_decode, for caller to track offset)
    """
    input_ids = mx.array(new_ids)[None]
    n_prompt = len(new_ids)

    # Prefill new tokens
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache_list)
    mx.eval(logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Sample first response token
    if temp == 0:
        token = mx.argmax(logits[:, -1, :], axis=-1)
    else:
        token = mx.random.categorical(logits[:, -1, :] / temp)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    response_ids = []
    t0 = time.perf_counter()

    for _ in range(max_tokens):
        tid = token.item()
        if eos_id is not None and tid == eos_id:
            break

        response_ids.append(tid)
        if stream:
            sys.stdout.write(tokenizer.decode([tid]))
            sys.stdout.flush()

        # Feed this token to model → get distribution for NEXT token
        logits = model(token.reshape(1, 1), cache=cache_list)
        mx.eval(logits)
        if temp == 0:
            token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            token = mx.random.categorical(logits[:, -1, :] / temp)

    decode_time = time.perf_counter() - t0
    n_decode = len(response_ids)

    return {
        "text": tokenizer.decode(response_ids),
        "n_decode": n_decode,
        "n_prompt": n_prompt,
        "prefill_ms": prefill_ms,
        "decode_tok_s": n_decode / decode_time if decode_time > 0 else 0,
        # Caller adds this to cache_token_count:
        # n_prompt new prefill tokens + n_decode response tokens fed as input
        "n_cache_added": n_prompt + n_decode,
    }


def cache_mb(cache_list):
    total = 0
    for c in cache_list:
        try:
            total += c.nbytes
        except Exception:
            pass
    return total / (1024 ** 2)


def mem_info():
    info = f"GPU peak: {mx.metal.get_peak_memory() / 2**30:.2f} GB"
    if HAS_PSUTIL:
        info += f" | RSS: {psutil.Process().memory_info().rss / 2**30:.2f} GB"
    return info


# ─── Benchmark ──────────────────────────────────────────────────────

def benchmark(model, tokenizer, args):
    prompt = "Explain the concept of KV cache compression for large language models in three detailed paragraphs."
    bits_list = [0, 8]   # 4-bit causes compound error across 28 layers
    max_tokens = 100

    print(f"\n{'=' * 70}")
    print("  TURBOQUANT-MLX BENCHMARK")
    print(f"{'=' * 70}")
    print(f"  Model:      {args.model}")
    print(f"  Layers:     {len(model.layers)}")
    print(f"  Head dim:   {model.head_dim}")
    print(f"  KV heads:   {model.n_kv_heads}")
    print(f"  Prompt:     {prompt[:50]}...")
    print(f"  Max tokens: {max_tokens}")

    results = []
    for bits in bits_list:
        name = "FP16 baseline" if bits == 0 else f"TQ {bits}-bit"
        print(f"\n{'─' * 60}")
        print(f"  [{name}]")

        mx.metal.reset_peak_memory()
        cache_list = make_turbo_cache(model, bits=bits)

        try:
            r = generate(model, tokenizer, prompt, cache_list, max_tokens)
            r["name"] = name
            r["cache_mb"] = cache_mb(cache_list)
            r["peak_gpu_gb"] = mx.metal.get_peak_memory() / 2**30
            results.append(r)

            print(f"  Prefill:  {r['prefill_ms']:.0f} ms")
            print(f"  Decode:   {r['decode_tok_s']:.1f} tok/s ({r['tokens']} tokens)")
            print(f"  Cache:    {r['cache_mb']:.1f} MB")
            print(f"  GPU peak: {r['peak_gpu_gb']:.2f} GB")
            print(f"  Output:   {r['text'][:150]}...")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(results) >= 2:
        print(f"\n{'=' * 70}")
        print("  SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Config':<16} {'tok/s':>8} {'Cache MB':>10} {'GPU GB':>8}")
        print(f"  {'─' * 46}")
        for r in results:
            print(f"  {r['name']:<16} {r['decode_tok_s']:>8.1f} {r['cache_mb']:>10.1f} {r['peak_gpu_gb']:>8.2f}")

        baseline = results[0]
        for r in results[1:]:
            if baseline["cache_mb"] > 0 and r["cache_mb"] > 0:
                comp = baseline["cache_mb"] / r["cache_mb"]
                speed = r["decode_tok_s"] / baseline["decode_tok_s"] if baseline["decode_tok_s"] > 0 else 0
                print(f"\n  {r['name']} vs baseline: {comp:.1f}x cache compression, {speed:.2f}x speed")


# ─── Interactive Chat ───────────────────────────────────────────────

def chat(model, tokenizer, args):
    print(f"\n{'=' * 70}")
    print("  TURBOQUANT-MLX — Interactive Chat")
    print(f"{'=' * 70}")
    bits = args.bits
    name = "FP16 baseline" if bits == 0 else f"TQ {bits}-bit"
    mode = "stateless (--reset-cache)" if args.reset_cache else "stateful (multi-turn)"
    print(f"  Model: {args.model}")
    print(f"  Cache: {name}  |  Mode: {mode}")
    print(f"  Type 'quit' to exit, 'reset' to clear context")
    print(f"{'=' * 70}\n")

    def _new_session():
        """Return a fresh cache + empty conversation state."""
        return make_turbo_cache(model, bits=bits), [], 0

    cache_list, messages, cache_token_count = _new_session()

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip():
            continue
        if prompt.strip().lower() == "quit":
            break
        if prompt.strip().lower() == "reset":
            cache_list, messages, cache_token_count = _new_session()
            print("[Context reset]\n")
            continue

        messages.append({"role": "user", "content": prompt})

        if args.reset_cache:
            # ── Stateless mode: fresh cache every turn ────────────────────────
            # Re-encode full conversation history each call (tool/script use)
            cache_list, _, cache_token_count = _new_session()
            full_ids = tokenizer.encode(_format_messages(tokenizer, messages))
            new_ids = full_ids  # process everything from scratch
        else:
            # ── Stateful mode: incremental prefill on existing cache ───────────
            # Only encode + prefill the tokens added since last turn
            full_ids = tokenizer.encode(_format_messages(tokenizer, messages))
            new_ids = full_ids[cache_token_count:]

        if not new_ids:
            print("[Warning: no new tokens to process]\n")
            continue

        print()
        r = generate_incremental(model, tokenizer, new_ids, cache_list,
                                 max_tokens=args.max_tokens, temp=args.temp, stream=True)
        print()

        # Track cache offset for next turn (only in stateful mode)
        if not args.reset_cache:
            cache_token_count += r["n_cache_added"]

        # Append clean response text to history (without EOS)
        messages.append({"role": "assistant", "content": r["text"]})

        c_mb = cache_mb(cache_list)
        turn_info = f"turn {len(messages) // 2}" if not args.reset_cache else "stateless"
        print(f"  [{r['n_decode']} tokens | {r['decode_tok_s']:.1f} tok/s | "
              f"cache: {c_mb:.1f} MB | {turn_info} | {mem_info()}]\n")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant-MLX")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--bits", type=int, default=8, choices=[0, 4, 8],
                        help="Cache quantization bits (0=FP16, 8=2x compression, 4=4x compression)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")
    parser.add_argument(
        "--reset-cache", action="store_true", default=False,
        help="Reset KV cache after every turn (stateless, single-turn mode). "
             "Use this when calling the model from scripts/tools. "
             "Default: off (stateful multi-turn — cache is kept across turns)."
    )
    args = parser.parse_args()

    print(f"\nLoading {args.model}...")
    model, tokenizer = load(args.model)
    print(f"Loaded: {len(model.layers)} layers, head_dim={model.head_dim}, n_kv_heads={model.n_kv_heads}")

    if args.benchmark:
        benchmark(model, tokenizer, args)
    else:
        chat(model, tokenizer, args)


if __name__ == "__main__":
    main()
