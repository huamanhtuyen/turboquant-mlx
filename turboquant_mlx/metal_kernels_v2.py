"""Optimized Metal kernels for TurboQuant — v2.

Key optimizations over v1:
1. Parallel WHT: threadgroup of D threads (one per dimension), barrier-synced butterfly
2. Parallel reduction for dot product (SIMD-friendly)
3. Single kernel launch per head group instead of per-position

v1: 1 thread per vector, serial butterfly → O(d log d) per thread
v2: D threads per vector, parallel butterfly → O(log d) per thread
"""

import mlx.core as mx
import math


# Parallel dequant kernel: one threadgroup per sequence position
# Each thread handles one dimension element
PARALLEL_DEQUANT_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];

    // Step 1: Codebook lookup (fully parallel — each thread one element)
    uint idx = indices[pos * dim + elem];
    T val = centroids[idx] * scale[0];

    // Store to threadgroup memory for butterfly
    threadgroup T shared[256];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: WHT butterfly — log2(dim) stages, all parallel
    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            // Even position: a + b
            uint j = block * 2 * h + offset;
            T a = shared[j];
            T b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Step 3: Apply signs, WHT scale, and vector norm
    T result = shared[elem] * scale[0] * signs[elem] * norms[pos];

    // Write output
    out[pos * dim + elem] = result;
"""

# Parallel fused Q@K^T: one threadgroup per (head, position)
# Computes dot product via parallel reduction
PARALLEL_FUSED_QK_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];

    uint kv_base = head * seq_len * dim + pos * dim;
    uint q_base = head * dim;

    // Step 1: Codebook lookup
    uint idx = indices[kv_base + elem];
    T val = centroids[idx] * scale[0];

    // Step 2: Parallel WHT butterfly in threadgroup memory
    threadgroup T shared[256];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j];
            T b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Step 3: Compute per-element product (dequant * query)
    T dequant_val = shared[elem] * scale[0] * signs[elem] * norms[head * seq_len + pos];
    T partial = dequant_val * query[q_base + elem];

    // Step 4: Parallel reduction for dot product
    shared[elem] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            shared[elem] += shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the final dot product
    if (elem == 0) {
        out[head * seq_len + pos] = shared[0];
    }
"""

_par_dequant_kernel = None
_par_fused_qk_kernel = None


def parallel_dequantize(
    indices: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
) -> mx.array:
    """Parallel TurboQuant dequantize — D threads per vector."""
    global _par_dequant_kernel
    if _par_dequant_kernel is None:
        _par_dequant_kernel = mx.fast.metal_kernel(
            name="turboquant_par_dequant",
            input_names=["indices", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=PARALLEL_DEQUANT_KERNEL,
        )

    seq_len = indices.shape[0]
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim], dtype=mx.uint32)

    outputs = _par_dequant_kernel(
        inputs=[indices.astype(mx.uint32), norms, centroids, signs, scale, dims_arr],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def parallel_fused_qk_scores(
    query: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
) -> mx.array:
    """Parallel fused Q@K^T — D threads per (head, position) via tree reduction."""
    global _par_fused_qk_kernel
    if _par_fused_qk_kernel is None:
        _par_fused_qk_kernel = mx.fast.metal_kernel(
            name="turboquant_par_fused_qk",
            input_names=["query", "indices", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=PARALLEL_FUSED_QK_KERNEL,
        )

    n_heads, seq_len = k_norms.shape
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, n_heads], dtype=mx.uint32)

    outputs = _par_fused_qk_kernel(
        inputs=[
            query.astype(mx.float32).reshape(n_heads * dim),
            k_indices.astype(mx.uint32).reshape(n_heads * seq_len * dim),
            k_norms.astype(mx.float32).reshape(n_heads * seq_len),
            centroids.astype(mx.float32),
            signs.astype(mx.float32),
            scale,
            dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, n_heads, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_heads * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, seq_len)
