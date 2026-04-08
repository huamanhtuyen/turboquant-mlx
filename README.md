# TurboQuant-MLX

KV cache compression cho MLX-LM, chạy trên Apple Silicon. Drop-in replacement cho `KVCache` mặc định, dùng `mx.quantize` built-in của MLX.

## Kết quả thực tế (Qwen2.5-7B-Instruct-4bit, M4 16GB)

| Config | tok/s | Cache MB | Output |
|--------|------:|--------:|--------|
| FP16 baseline | 22.2 | 7.9 MB | ✅ Coherent |
| **TQ 8-bit** | **22.1** | **4.5 MB** | ✅ Identical |

- **1.8x cache compression**, 0% speed loss
- Prefill nhanh hơn ~42% (477ms → 277ms)

---

## Cài đặt

```bash
# Clone repo
git clone https://github.com/arozanov/turboquant-mlx
cd turboquant-mlx

# Tạo venv với Python 3.11
python3.11 -m venv tq29
source tq29/bin/activate

# Cài dependencies
pip install mlx mlx-lm psutil
```

---

## Sử dụng

### Interactive chat

```bash
# Chat với 8-bit cache (mặc định, recommended)
python run.py

# Chat với FP16 (baseline, không nén)
python run.py --bits 0

# Đổi model
python run.py --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

Trong chat:
- Gõ câu hỏi rồi Enter
- Gõ `reset` để xóa context
- Gõ `quit` để thoát

### Benchmark

```bash
# So sánh FP16 vs TQ 8-bit
python run.py --benchmark

# Benchmark với model nhỏ hơn
python run.py --benchmark --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

### Options

```
--model       HuggingFace repo (default: mlx-community/Qwen2.5-7B-Instruct-4bit)
--bits        0 = FP16, 8 = 8-bit compressed (default: 8)
--max-tokens  Số token tối đa mỗi lượt (default: 200)
--temp        Temperature (0 = greedy, default: 0.0)
--benchmark   Chạy benchmark so sánh thay vì chat
```

---

## Dùng trong code

```python
from mlx_lm import load
from turbo_cache import TurboQuantCache

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Tạo cache list (thay cho make_kv_caches mặc định)
cache = [
    TurboQuantCache(model.head_dim, model.n_kv_heads, bits=8)
    for _ in range(len(model.layers))
]

# Dùng như KVCache bình thường
import mlx.core as mx
tokens = mx.array(tokenizer.encode("Hello!"))[None]
logits = model(tokens, cache=cache)
```

### TurboQuantCache API

```python
TurboQuantCache(
    head_dim,        # int hoặc (k_dim, v_dim)
    n_kv_heads,      # số KV heads của model
    bits=8,          # 0=FP16, 8=compressed (recommended)
    group_size=64,   # nhóm quantization (32, 64, 128)
)
```

| Method/Property | Mô tả |
|----------------|-------|
| `update_and_fetch(k, v)` | Lưu và trả về toàn bộ K/V (giống KVCache) |
| `offset` | Số tokens đã cached |
| `nbytes` | Dung lượng cache hiện tại (bytes) |
| `state` | Tuple (k_buf, v_buf) — dùng cho serialization |

---

## Tại sao không dùng 4-bit?

`mx.quantize` với `bits=4` được thiết kế cho **weight matrices** (tĩnh). KV cache là dynamic — mỗi token từ model có phân phối khác nhau. Với 28 layers, quantization error tích lũy qua các lớp làm sai lệch output (~25 điểm logit diff vs ~0.8 với 8-bit).

MLX's own `QuantizedKVCache` cũng dùng 8-bit vì lý do này.

---

## Yêu cầu hệ thống

- macOS 14+ (Apple Silicon M1/M2/M3/M4)
- Python 3.11
- `mlx >= 0.18`, `mlx-lm >= 0.18`
- RAM: ≥ 16GB để chạy 7B model

---

## Cấu trúc file

```
turboquant-mlx/
├── turbo_cache.py      # TurboQuantCache — cache implementation chính
├── run.py              # Script chạy: chat + benchmark
├── README.md           # Tài liệu này
└── tq29/               # Python venv (không commit)
```
