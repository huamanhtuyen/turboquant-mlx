"""Speed comparison: Python vs Metal dequant in cache context."""

import sys
sys.path.insert(0, "/Users/antonrozanov/Projects/turboquant-money/turboquant-mlx")

import mlx.core as mx
import time
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.metal_kernels import fused_dequantize


def bench_dequant(seq_len, dim=128, bits=3, n_iters=50):
    """Compare Python vs Metal dequant at given sequence length."""
    pq = PolarQuantizer(dim, bits=bits)
    x = mx.random.normal(shape=(seq_len, dim))
    indices, norms = pq.quantize(x)
    mx.eval(indices, norms)

    # Warmup
    for _ in range(5):
        _ = pq.dequantize(indices, norms)
        mx.eval(_)
        _ = fused_dequantize(indices, norms, pq.centroids, pq.signs, dim)
        mx.eval(_)

    # Python path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out_py = pq.dequantize(indices, norms)
        mx.eval(out_py)
    py_ms = (time.perf_counter() - t0) / n_iters * 1000

    # Metal path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out_metal = fused_dequantize(indices, norms, pq.centroids, pq.signs, dim)
        mx.eval(out_metal)
    metal_ms = (time.perf_counter() - t0) / n_iters * 1000

    speedup = py_ms / metal_ms if metal_ms > 0 else 0
    return py_ms, metal_ms, speedup


def bench_full_cache_cycle(seq_len, n_heads=32, dim=128, bits=3, n_iters=20):
    """Simulate decode: quantize 1 token + dequant full cache."""
    from turboquant_mlx.cache import TurboQuantKVCache

    # Fill cache to seq_len
    cache_metal = TurboQuantKVCache(bits=bits)
    B = 1
    keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    cache_metal.update_and_fetch(keys, vals)
    mx.eval(cache_metal.k_indices)

    # Simulate decode steps
    new_k = mx.random.normal(shape=(B, n_heads, 1, dim))
    new_v = mx.random.normal(shape=(B, n_heads, 1, dim))
    mx.eval(new_k, new_v)

    # Warmup
    for _ in range(3):
        # Reset offset to simulate repeated decode
        cache_metal.offset = seq_len
        out_k, out_v = cache_metal.update_and_fetch(new_k, new_v)
        mx.eval(out_k, out_v)

    # Benchmark: Metal cache decode
    times = []
    for _ in range(n_iters):
        cache_metal.offset = seq_len
        t0 = time.perf_counter()
        out_k, out_v = cache_metal.update_and_fetch(new_k, new_v)
        mx.eval(out_k, out_v)
        times.append(time.perf_counter() - t0)
    metal_ms = sum(times) / len(times) * 1000

    return metal_ms


def main():
    print("=" * 60)
    print("TurboQuant Speed Benchmark: Python vs Metal")
    print("=" * 60)

    print("\n--- Dequant Only (single head) ---")
    print(f"{'seq_len':>10} {'Python ms':>12} {'Metal ms':>12} {'Speedup':>10}")
    print("-" * 48)
    for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
        py_ms, metal_ms, speedup = bench_dequant(seq_len)
        print(f"{seq_len:>10} {py_ms:>12.3f} {metal_ms:>12.3f} {speedup:>9.2f}x")

    print("\n--- Full Cache Decode (32 heads, d=128, 3-bit) ---")
    print(f"{'seq_len':>10} {'Metal ms':>12} {'Effective tok/s':>16}")
    print("-" * 42)
    for seq_len in [128, 512, 1024, 2048, 4096]:
        metal_ms = bench_full_cache_cycle(seq_len, n_heads=32, dim=128, bits=3)
        tok_s = 1000.0 / metal_ms if metal_ms > 0 else 0
        print(f"{seq_len:>10} {metal_ms:>12.3f} {tok_s:>16.1f}")


if __name__ == "__main__":
    main()
