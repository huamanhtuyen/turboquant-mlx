"""Metal kernels v4: Pre-rotated query optimization.

Key insight: instead of inverse-WHT on every cached K vector,
apply forward-WHT once to the query:

  dot(Q, dequant(K)) = (norm/D) * dot(WHT(signs*Q), codebook[indices])

This eliminates the O(d log d) WHT from the inner loop.
The kernel just does codebook lookups + multiply-accumulate — trivially parallel.

Speedup: O(seq_len * d * log d) → O(seq_len * d) per decode step.
"""

import mlx.core as mx
import math
from turboquant_mlx.rotation import randomized_hadamard_transform


# Fused Q@K^T with pre-rotated query
# No WHT in the inner loop — just codebook lookup + dot product
PREROT_FUSED_QK_KERNEL = """
    uint pos = thread_position_in_grid.x;
    uint head = thread_position_in_grid.y;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint bit_mask = (1u << bits) - 1u;

    uint kv_base = head * seq_len * packed_dim + pos * packed_dim;
    uint q_base = head * dim;
    T vec_norm = norms[head * seq_len + pos];
    T norm_scale = vec_norm * inv_d[0];

    // Simple dot product: Q_rot[d] * codebook[indices[d]]
    T dot = 0;
    for (uint i = 0; i < dim; i++) {
        uint word_idx = i / vals_per_word;
        uint pos_in_word = i % vals_per_word;
        uint word = packed[kv_base + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;

        dot += q_rot[q_base + i] * centroids[idx];
    }

    out[head * seq_len + pos] = dot * norm_scale;
"""

# Dequant kernel with pre-rotated approach (for V dequant)
# Still needs full dequant for V since we need the actual vectors
PREROT_DEQUANT_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint bit_mask = (1u << bits) - 1u;

    // Extract index from packed
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[pos * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    T val = centroids[idx] * scale[0];

    // Parallel WHT butterfly
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

    T result = shared[elem] * scale[0] * signs[elem] * norms[pos];
    out[pos * dim + elem] = result;
"""

_prerot_qk_kernel = None
_prerot_dequant_kernel = None


def prerotate_query(query: mx.array, signs: mx.array) -> mx.array:
    """Apply forward rotation to query: Q_rot = WHT(signs * Q).

    Done once per decode step per head. O(d log d) but only 1 vector.
    """
    return randomized_hadamard_transform(query.astype(mx.float32), signs)


def prerot_fused_qk_scores(
    q_rot: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Fused Q@K^T with pre-rotated query. No WHT in inner loop."""
    global _prerot_qk_kernel
    if _prerot_qk_kernel is None:
        _prerot_qk_kernel = mx.fast.metal_kernel(
            name="tq_prerot_fused_qk",
            input_names=["q_rot", "packed", "norms", "centroids", "inv_d", "dims"],
            output_names=["out"],
            source=PREROT_FUSED_QK_KERNEL,
        )

    n_heads, seq_len = k_norms.shape
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    inv_d = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, n_heads, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _prerot_qk_kernel(
        inputs=[
            q_rot.astype(mx.float32).reshape(n_heads * dim),
            k_packed.astype(mx.uint32).reshape(n_heads * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_heads * seq_len),
            centroids.astype(mx.float32),
            inv_d, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(seq_len, n_heads, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(n_heads * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, seq_len)


def prerot_packed_dequantize(
    packed: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Dequantize from packed storage (used for V, which needs full vectors)."""
    global _prerot_dequant_kernel
    if _prerot_dequant_kernel is None:
        _prerot_dequant_kernel = mx.fast.metal_kernel(
            name="tq_prerot_dequant",
            input_names=["packed", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=PREROT_DEQUANT_KERNEL,
        )

    seq_len = norms.shape[0]
    p_dim = packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _prerot_dequant_kernel(
        inputs=[packed.astype(mx.uint32).reshape(-1), norms, centroids, signs, scale, dims_arr],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
