"""Test fused attention: correctness and speed."""

import sys
sys.path.insert(0, "/Users/antonrozanov/Projects/turboquant-money/turboquant-mlx")

import mlx.core as mx
import time
from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.fused_attention import fused_qk_scores, turboquant_attention
from turboquant_mlx.quantizer import PolarQuantizer


def test_fused_qk_correctness():
    """Fused Q@K^T matches naive dequant+matmul."""
    dim = 128
    n_heads = 4
    seq_len = 64
    bits = 3

    pq = PolarQuantizer(dim, bits=bits)

    # Simulate cached keys
    keys = mx.random.normal(shape=(n_heads, seq_len, dim))
    k_indices = mx.zeros((n_heads, seq_len, dim), dtype=mx.uint8)
    k_norms = mx.zeros((n_heads, seq_len), dtype=mx.float32)

    for h in range(n_heads):
        idx, nrm = pq.quantize(keys[h])
        k_indices[h] = idx
        k_norms[h] = nrm

    query = mx.random.normal(shape=(n_heads, dim))

    # Naive: dequant then matmul
    naive_scores = mx.zeros((n_heads, seq_len))
    for h in range(n_heads):
        k_deq = pq.dequantize(k_indices[h], k_norms[h])
        naive_scores[h] = k_deq @ query[h]

    # Fused
    fused_scores = fused_qk_scores(query, k_indices, k_norms, pq.centroids, pq.signs, dim)
    mx.eval(naive_scores, fused_scores)

    error = mx.abs(naive_scores - fused_scores).max().item()
    corr = (mx.sum(naive_scores * fused_scores) /
            (mx.linalg.norm(naive_scores.reshape(-1)) * mx.linalg.norm(fused_scores.reshape(-1)) + 1e-8)).item()

    print(f"  Q@K^T: max_error={error:.6f}, correlation={corr:.6f}")
    assert corr > 0.999, f"Fused Q@K^T diverges: {corr}"


def test_full_attention_correctness():
    """Full fused attention matches naive path."""
    B, n_heads, dim = 1, 4, 128
    seq_len = 32
    bits = 3

    # Build cache
    cache_naive = TurboQuantKVCache(bits=bits, fused=False)
    cache_fused = TurboQuantKVCache(bits=bits, fused=False)  # both non-fused for fair comparison

    keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))

    # Fill both caches identically
    cache_naive.update_and_fetch(keys, vals)
    cache_fused.update_and_fetch(keys, vals)
    mx.eval(cache_naive.k_indices, cache_fused.k_indices)

    # Single query
    query = mx.random.normal(shape=(B, n_heads, 1, dim))
    scale = 1.0 / (dim ** 0.5)

    # Naive: dequant all, standard attention
    k_deq, v_deq = cache_naive._metal_dequantize(
        cache_naive.k_indices, cache_naive.k_norms,
        cache_naive._k_quantizer, dim, B, n_heads, seq_len, mx.float32,
    ), cache_naive._metal_dequantize(
        cache_naive.v_indices, cache_naive.v_norms,
        cache_naive._v_quantizer, dim, B, n_heads, seq_len, mx.float32,
    )
    naive_out = mx.fast.scaled_dot_product_attention(query, k_deq, v_deq, scale=scale)

    # Fused path
    fused_out = turboquant_attention(query, cache_fused, scale, mask=None)
    mx.eval(naive_out, fused_out)

    cos = (mx.sum(naive_out * fused_out) /
           (mx.linalg.norm(naive_out.reshape(-1)) * mx.linalg.norm(fused_out.reshape(-1)) + 1e-8)).item()

    print(f"  Full attention: cosine={cos:.6f}")
    assert cos > 0.99, f"Fused attention diverges: {cos}"


def test_fused_speed():
    """Benchmark fused vs dequant+attention decode speed."""
    B, n_heads, dim = 1, 32, 128
    bits = 3

    for seq_len in [256, 1024, 2048, 4096]:
        # Fill cache
        cache = TurboQuantKVCache(bits=bits, fused=False)
        keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
        vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))
        cache.update_and_fetch(keys, vals)
        mx.eval(cache.k_indices)

        query = mx.random.normal(shape=(B, n_heads, 1, dim))
        scale = 1.0 / (dim ** 0.5)
        mx.eval(query)

        n_iters = 20

        # Warmup
        for _ in range(3):
            k_d = cache._metal_dequantize(cache.k_indices, cache.k_norms, cache._k_quantizer, dim, B, n_heads, seq_len, mx.float16)
            v_d = cache._metal_dequantize(cache.v_indices, cache.v_norms, cache._v_quantizer, dim, B, n_heads, seq_len, mx.float16)
            out = mx.fast.scaled_dot_product_attention(query, k_d, v_d, scale=scale)
            mx.eval(out)

            out2 = turboquant_attention(query, cache, scale, mask=None)
            mx.eval(out2)

        # Naive: dequant + SDPA
        t0 = time.perf_counter()
        for _ in range(n_iters):
            k_d = cache._metal_dequantize(cache.k_indices, cache.k_norms, cache._k_quantizer, dim, B, n_heads, seq_len, mx.float16)
            v_d = cache._metal_dequantize(cache.v_indices, cache.v_norms, cache._v_quantizer, dim, B, n_heads, seq_len, mx.float16)
            out = mx.fast.scaled_dot_product_attention(query, k_d, v_d, scale=scale)
            mx.eval(out)
        naive_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Fused
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out2 = turboquant_attention(query, cache, scale, mask=None)
            mx.eval(out2)
        fused_ms = (time.perf_counter() - t0) / n_iters * 1000

        speedup = naive_ms / fused_ms if fused_ms > 0 else 0
        print(f"  seq={seq_len:>5}: naive={naive_ms:.2f}ms, fused={fused_ms:.2f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    tests = [
        ("Fused Q@K^T correctness", test_fused_qk_correctness),
        ("Full attention correctness", test_full_attention_correctness),
        ("Fused decode speed", test_fused_speed),
    ]

    print("=" * 55)
    print("TurboQuant Fused Attention Tests")
    print("=" * 55)

    for name, test in tests:
        print(f"\n[{name}]")
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 55}")
