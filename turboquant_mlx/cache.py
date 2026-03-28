"""TurboQuantKVCache: PolarQuant KV cache compression for MLX.

Drop-in replacement for mlx_lm's KVCache. Compresses KV cache vectors
using PolarQuant (randomized Hadamard rotation + Lloyd-Max scalar quantization).

Stores indices in bit-packed uint32 format for real memory savings:
  3-bit → 4.6x compression vs fp16 (10 values per uint32)

Usage:
    from turboquant_mlx import TurboQuantKVCache
    cache = [TurboQuantKVCache(bits=3) for _ in range(num_layers)]
"""

import mlx.core as mx
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.packing import pack_indices, unpack_indices, packed_dim, VALS_PER_WORD
from turboquant_mlx.metal_kernels_v3 import packed_dequantize
from turboquant_mlx.metal_quantize import fused_quantize, dequant_fp16
from turboquant_mlx.fused_attention import turboquant_attention


class TurboQuantKVCache:
    """TurboQuant KV cache with PolarQuant compression and bit-packed storage."""

    step = 256

    def __init__(self, bits: int = 3, seed: int = 42, fused: bool = True):
        self.quant_bits = bits
        self.seed = seed
        self.offset = 0
        self.fused = fused

        # Packed storage: uint32 (B, H, S, packed_dim) + float32 norms (B, H, S)
        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None

        # Pre-allocated decode buffer: running dequantized cache for fast decode
        # Only dequant new tokens, write in-place → O(1) per step, zero allocation
        self._k_deq_buf = None  # (B, H, alloc_size, k_dim)
        self._v_deq_buf = None  # (B, H, alloc_size, v_dim)
        self._deq_offset = 0    # how many tokens are filled
        self._deq_alloc = 0     # how many tokens are allocated

        self._k_quantizer = None
        self._v_quantizer = None
        self._k_dim = None
        self._v_dim = None
        self._k_packed_dim = None
        self._v_packed_dim = None

    def _ensure_quantizer(self, k_dim: int, v_dim: int):
        if self._k_quantizer is None:
            self._k_quantizer = PolarQuantizer(k_dim, bits=self.quant_bits, seed=self.seed)
            self._k_dim = k_dim
            self._k_packed_dim = packed_dim(k_dim, self.quant_bits)
        if self._v_quantizer is None:
            self._v_quantizer = PolarQuantizer(v_dim, bits=self.quant_bits, seed=self.seed + 1)
            self._v_dim = v_dim
            self._v_packed_dim = packed_dim(v_dim, self.quant_bits)

    def _ensure_storage(self, B, H, num_new, k_dim, v_dim):
        prev = self.offset
        needed = prev + num_new

        if self.k_packed is None or needed > self.k_packed.shape[2]:
            n_steps = ((needed + self.step - 1) // self.step) * self.step

            new_kp = mx.zeros((B, H, n_steps, self._k_packed_dim), dtype=mx.uint32)
            new_kn = mx.zeros((B, H, n_steps), dtype=mx.float32)
            new_vp = mx.zeros((B, H, n_steps, self._v_packed_dim), dtype=mx.uint32)
            new_vn = mx.zeros((B, H, n_steps), dtype=mx.float32)

            if self.k_packed is not None:
                old_kp = self.k_packed[..., :prev, :]
                old_kn = self.k_norms[..., :prev]
                old_vp = self.v_packed[..., :prev, :]
                old_vn = self.v_norms[..., :prev]
                self.k_packed = mx.concatenate([old_kp, new_kp], axis=2)
                self.k_norms = mx.concatenate([old_kn, new_kn], axis=2)
                self.v_packed = mx.concatenate([old_vp, new_vp], axis=2)
                self.v_norms = mx.concatenate([old_vn, new_vn], axis=2)
            else:
                self.k_packed = new_kp
                self.k_norms = new_kn
                self.v_packed = new_vp
                self.v_norms = new_vn

    def _packed_dequant(self, packed, norms, quantizer, dim, B, H, total, out_dtype):
        """Dequantize directly from packed uint32 via Metal kernel — no Python unpack."""
        flat_packed = packed[..., :total, :].reshape(-1, packed.shape[-1])
        flat_norms = norms[..., :total].reshape(-1)

        flat_out = packed_dequantize(
            flat_packed, flat_norms,
            quantizer.centroids, quantizer.signs,
            dim, self.quant_bits,
        )
        return flat_out.reshape(B, H, total, dim).astype(out_dtype)

    def _get_unpacked_indices(self, packed, dim, total):
        """Get unpacked uint8 indices for fused attention path."""
        flat_packed = packed[..., :total, :].reshape(-1, packed.shape[-1])
        return unpack_indices(flat_packed, self.quant_bits, dim)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, k_dim = keys.shape
        v_dim = values.shape[3]

        self._ensure_quantizer(k_dim, v_dim)
        self._ensure_storage(B, H, S, k_dim, v_dim)

        prev = self.offset

        # Fused Metal quantize: raw vectors → packed uint32 + norms in one kernel
        k_flat = keys.reshape(-1, k_dim)
        k_pk_flat, k_nrm = fused_quantize(
            k_flat, self._k_quantizer.signs,
            self._k_quantizer.boundaries, k_dim, self.quant_bits)
        k_pk = k_pk_flat.reshape(B, H, S, self._k_packed_dim)

        v_flat = values.reshape(-1, v_dim)
        v_pk_flat, v_nrm = fused_quantize(
            v_flat, self._v_quantizer.signs,
            self._v_quantizer.boundaries, v_dim, self.quant_bits)
        v_pk = v_pk_flat.reshape(B, H, S, self._v_packed_dim)

        # Store packed
        self.k_packed[..., prev:prev + S, :] = k_pk
        self.k_norms[..., prev:prev + S] = k_nrm.reshape(B, H, S)
        self.v_packed[..., prev:prev + S, :] = v_pk
        self.v_norms[..., prev:prev + S] = v_nrm.reshape(B, H, S)

        self.offset += S
        total = self.offset

        # --- Incremental decode: dequant only new tokens, write in-place ---
        if S <= 4 and self._v_deq_buf is not None and self._deq_offset == prev:
            # Ensure buffer has space
            if total > self._deq_alloc:
                new_alloc = ((total + self.step - 1) // self.step) * self.step
                if self._k_deq_buf is not None:
                    k_ext = mx.zeros((B, H, new_alloc - self._deq_alloc, k_dim), dtype=keys.dtype)
                    self._k_deq_buf = mx.concatenate([self._k_deq_buf[..., :self._deq_offset, :], k_ext], axis=2)
                v_ext = mx.zeros((B, H, new_alloc - self._deq_alloc, v_dim), dtype=values.dtype)
                self._v_deq_buf = mx.concatenate([self._v_deq_buf[..., :self._deq_offset, :], v_ext], axis=2)
                self._deq_alloc = new_alloc

            # Always dequant new V token (needed for weighted sum)
            new_v_pk = v_pk.reshape(-1, v_pk.shape[-1])
            new_v_deq = dequant_fp16(
                new_v_pk, v_nrm,
                self._v_quantizer.centroids, self._v_quantizer.signs,
                v_dim, self.quant_bits,
            ).reshape(B, H, S, v_dim)
            self._v_deq_buf[..., prev:total, :] = new_v_deq

            if self.fused:
                # Fused: skip K dequant, fused kernel reads packed K directly
                self._deq_offset = total
                dummy_k = mx.zeros((B, H, total, k_dim), dtype=keys.dtype)
                return dummy_k, self._v_deq_buf[..., :total, :]
            else:
                # Non-fused: also dequant new K token
                new_k_pk = k_pk.reshape(-1, k_pk.shape[-1])
                new_k_deq = dequant_fp16(
                    new_k_pk, k_nrm,
                    self._k_quantizer.centroids, self._k_quantizer.signs,
                    k_dim, self.quant_bits,
                ).reshape(B, H, S, k_dim)
                self._k_deq_buf[..., prev:total, :] = new_k_deq
                self._deq_offset = total
                return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

        # Full dequant (prefill or first decode step)
        all_keys = self._packed_dequant(
            self.k_packed, self.k_norms, self._k_quantizer, k_dim, B, H, total, keys.dtype)
        all_vals = self._packed_dequant(
            self.v_packed, self.v_norms, self._v_quantizer, v_dim, B, H, total, values.dtype)

        # Init pre-allocated decode buffer
        alloc = ((total + self.step - 1) // self.step) * self.step
        self._k_deq_buf = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
        self._v_deq_buf = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
        self._k_deq_buf[..., :total, :] = all_keys
        self._v_deq_buf[..., :total, :] = all_vals
        self._deq_offset = total
        self._deq_alloc = alloc

        return all_keys, all_vals

    # --- For fused attention: expose unpacked indices ---

    @property
    def k_indices(self):
        """Unpacked K indices for fused attention. Shape: (B, H, offset, k_dim)."""
        if self.k_packed is None:
            return None
        B, H = self.k_packed.shape[:2]
        flat = self.k_packed[..., :self.offset, :].reshape(-1, self._k_packed_dim)
        idx = unpack_indices(flat, self.quant_bits, self._k_dim)
        return idx.reshape(B, H, self.offset, self._k_dim)

    @property
    def v_indices(self):
        """Unpacked V indices for fused attention. Shape: (B, H, offset, v_dim)."""
        if self.v_packed is None:
            return None
        B, H = self.v_packed.shape[:2]
        flat = self.v_packed[..., :self.offset, :].reshape(-1, self._v_packed_dim)
        idx = unpack_indices(flat, self.quant_bits, self._v_dim)
        return idx.reshape(B, H, self.offset, self._v_dim)

    def empty(self):
        return self.k_packed is None

    @property
    def nbytes(self):
        if self.k_packed is None:
            return 0
        pk_bytes = (self.k_packed[..., :self.offset, :].nbytes +
                    self.v_packed[..., :self.offset, :].nbytes)
        nrm_bytes = (self.k_norms[..., :self.offset].nbytes +
                     self.v_norms[..., :self.offset].nbytes)
        return pk_bytes + nrm_bytes

    @property
    def uncompressed_nbytes(self):
        if self.k_packed is None:
            return 0
        B, H = self.k_packed.shape[:2]
        return self.offset * (self._k_dim + self._v_dim) * B * H * 2

    @property
    def compression_ratio(self):
        if self.nbytes == 0:
            return 0
        return self.uncompressed_nbytes / self.nbytes

    @property
    def state(self):
        if self.k_packed is None:
            return []
        return [
            self.k_packed[..., :self.offset, :],
            self.k_norms[..., :self.offset],
            self.v_packed[..., :self.offset, :],
            self.v_norms[..., :self.offset],
        ]

    @state.setter
    def state(self, v):
        if not v:
            return
        self.k_packed, self.k_norms, self.v_packed, self.v_norms = v
        self.offset = self.k_packed.shape[2]

    @property
    def meta_state(self):
        return f"{self.offset},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0}"

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self.offset = int(parts[0])
        self.quant_bits = int(parts[1])
        self.seed = int(parts[2])
        self._k_dim = int(parts[3]) or None
        self._v_dim = int(parts[4]) or None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(*args, offset=self.offset, **kwargs)
