"""Test fused Metal kernels for TurboQuant."""

import mlx.core as mx
import math
import sys
import time

sys.path.insert(0, "/Users/antonrozanov/Projects/turboquant-money/turboquant-mlx")

from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.metal_kernels import fused_dequantize, fused_attention_scores


def test_fused_dequant():
    """Fused Metal dequant matches Python dequant."""
    dim = 128
    seq_len = 64
    bits = 3
    pq = PolarQuantizer(dim, bits=bits)

    x = mx.random.normal(shape=(seq_len, dim))
    indices, norms = pq.quantize(x)

    # Python path
    x_py = pq.dequantize(indices, norms)

    # Metal path
    x_metal = fused_dequantize(indices, norms, pq.centroids, pq.signs, dim)

    mx.eval(x_py, x_metal)

    error = mx.abs(x_py - x_metal).max().item()
    cos_sim = (mx.sum(x_py * x_metal) / (mx.linalg.norm(x_py.reshape(-1)) * mx.linalg.norm(x_metal.reshape(-1)) + 1e-8)).item()

    print(f"  Fused dequant: max_error={error:.6f}, cosine_sim={cos_sim:.6f}")
    assert cos_sim > 0.99, f"Metal dequant diverges: cos_sim={cos_sim}"


def test_fused_attention():
    """Fused attention scores match naive compute."""
    dim = 128
    seq_len = 256
    bits = 3
    pq = PolarQuantizer(dim, bits=bits)

    # Simulate cached keys
    keys = mx.random.normal(shape=(seq_len, dim))
    k_indices, k_norms = pq.quantize(keys)

    # Single query
    query = mx.random.normal(shape=(dim,))

    # Naive: dequant then dot
    keys_deq = pq.dequantize(k_indices, k_norms)
    scores_naive = keys_deq @ query  # (seq_len,)

    # Fused: indices → scores directly
    scores_fused = fused_attention_scores(
        query, k_indices, k_norms, pq.centroids, pq.signs, dim
    )

    mx.eval(scores_naive, scores_fused)

    error = mx.abs(scores_naive - scores_fused).max().item()
    corr = mx.sum(scores_naive * scores_fused).item() / (
        mx.linalg.norm(scores_naive).item() * mx.linalg.norm(scores_fused).item() + 1e-8
    )

    print(f"  Fused attention: max_error={error:.6f}, correlation={corr:.6f}")
    assert corr > 0.99, f"Fused attention diverges: corr={corr}"


def test_fused_speed():
    """Benchmark fused vs naive dequant speed."""
    dim = 128
    seq_len = 2048
    bits = 3
    pq = PolarQuantizer(dim, bits=bits)

    keys = mx.random.normal(shape=(seq_len, dim))
    k_indices, k_norms = pq.quantize(keys)
    query = mx.random.normal(shape=(dim,))

    mx.eval(k_indices, k_norms, query)

    # Warmup
    for _ in range(3):
        _ = pq.dequantize(k_indices, k_norms)
        mx.eval(_)
        _ = fused_attention_scores(query, k_indices, k_norms, pq.centroids, pq.signs, dim)
        mx.eval(_)

    # Naive path: dequant + matmul
    t0 = time.perf_counter()
    for _ in range(20):
        deq = pq.dequantize(k_indices, k_norms)
        scores = deq @ query
        mx.eval(scores)
    naive_ms = (time.perf_counter() - t0) / 20 * 1000

    # Fused path
    t0 = time.perf_counter()
    for _ in range(20):
        scores = fused_attention_scores(query, k_indices, k_norms, pq.centroids, pq.signs, dim)
        mx.eval(scores)
    fused_ms = (time.perf_counter() - t0) / 20 * 1000

    speedup = naive_ms / fused_ms if fused_ms > 0 else 0
    print(f"  seq_len={seq_len}: naive={naive_ms:.2f}ms, fused={fused_ms:.2f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    tests = [
        ("Fused dequant correctness", test_fused_dequant),
        ("Fused attention correctness", test_fused_attention),
        ("Fused speed benchmark", test_fused_speed),
    ]

    print("=" * 50)
    print("TurboQuant Metal Kernel Tests")
    print("=" * 50)

    for name, test in tests:
        print(f"\n[{name}]")
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 50}")
