# Flash Attention (Rust: CPU + Metal GPU)

[中文](README_CN.md) | English

A learning-oriented **Flash Attention / Scaled Dot-Product Attention** project in Rust with two backends:

- **CPU**: pure Rust, zero core dependencies; naive reference, the original tiled implementation, a cached variant, and a multi-threaded variant; optional PyO3/Python binding.
- **GPU**: macOS **Metal** compute backend (MSL kernels + Rust driver): subwarp-per-query mapping, K/V tiles in threadgroup memory, 8-key sub-tile online softmax, causal masking; plus a materialized `S×S` naive baseline.

It ships with a reproducible **CPU vs GPU comparison** covering latency, throughput, numerical error, memory, CPU thread scaling, tile tuning, and launch/transfer overhead.

> [!WARNING]
> This is a personal learning project; please do not use it in production.

## Features

- Pure Rust CPU implementation with block-wise online softmax; never materializes the `S×S` attention matrix
- Multi-batch / multi-head inputs, causal masking, configurable scale and block size
- CPU optimizations: cached tile scores, scratch reuse, row-level parallelism with `std::thread` (no extra dependency)
- Metal Flash Attention: 4/8 lanes per query, Q/O in registers, 32 KiB threadgroup tiles, MSL specialized per head dimension
- Metal naive baseline (`scores → softmax → P·V`) to quantify the `O(S²)` memory wall
- Optional PyO3 Python extension (CPU path)
- Benchmark harness plus chart/table generation

## Project layout

```
flash-attention/
├── Cargo.toml
├── README.md / README_CN.md
├── GPU_COMPARISON.md          # Chinese: full CPU vs GPU comparison
├── GPU_COMPARISON_EN.md       # English summary
├── ARCHITECTURE.md
├── src/
│   ├── attention.rs           # CPU core: Tensor4, naive and tiled online softmax
│   ├── cpu_opt.rs             # cached + multi-threaded CPU variants
│   ├── gpu/
│   │   ├── mod.rs             # Metal device/pipelines/buffers/dispatch/timing
│   │   └── attention.metal    # four MSL kernels
│   ├── lib.rs                 # exports + optional Python bindings
│   ├── main.rs                # CPU/GPU demo
│   └── bin/compare.rs         # comparison harness
├── scripts/report.py          # CSV -> charts + markdown tables
├── results/                   # raw data, charts, tables (generated)
└── test.py                    # Python smoke test
```

## Quick start

### 1. CPU: build and test

```bash
cargo build --release
cargo test
```

Run the demo (with GPU when the feature is enabled):

```bash
cargo run --release --features metal
```

### 2. GPU (macOS + Metal)

Requirements: macOS with Metal 2.1+ (validated on macOS 26 / Apple M4), Rust 1.85+. MSL is compiled at runtime, so no `build.rs` is needed, but a working Metal runtime is required.

```bash
cargo test --features metal
```

Rust API:

```rust
use flash_attention::gpu::{AttnOptions, GpuAttention};
use flash_attention::make_qkv_example;

let (q, k, v) = make_qkv_example(1, 8, 1024, 64, 2024);
let gpu = GpuAttention::new().expect("Metal device");

let out = gpu
    .flash_attention(
        &q,
        &k,
        &v,
        AttnOptions {
            causal: true,
            block_size: 32,   // multiple of 16
            query_block: 32,  // query_block * lanes % 32 == 0
            ..Default::default()
        },
    )
    .expect("attention");
```

GPU constraints:

- head dim must be a multiple of 8 and ≤ 128 (64/128 are typical);
- `block_size` must be a multiple of 16 and ≤ `max_block_size(dim)` (64 for D=64, 32 for D=128);
- `query_block * lanes` must be a multiple of 32 (lanes = 4 for D≤64, 8 for D=128);
- macOS only; the Python binding currently uses the CPU path.

### 3. CPU vs GPU comparison

```bash
# Full suite (about 2-3 minutes)
cargo run --release --features metal --bin compare

# Quick smoke run
cargo run --release --features metal --bin compare -- --quick

# Charts + markdown tables
python3 -m venv .venv
.venv/bin/pip install pandas matplotlib
MPLCONFIGDIR=$PWD/.matplotlib-cache .venv/bin/python scripts/report.py
```

Outputs: `results/timings.csv`, `results/blocksize.csv`, `results/threads.csv`, `results/transfer.csv`, `results/startup.csv`, `results/meta.txt`, `results/tables.md`, `results/charts/*.png`.

### 4. Python (CPU)

```bash
python3 -m pip install numpy maturin
maturin build --release --features python
python3 -m pip install --force-reinstall target/wheels/*.whl
python3 test.py
```

```python
import numpy as np
import flash_attention

q = np.random.randn(2, 3, 5, 4).astype(np.float32)
out = flash_attention.flash_attention_py(q, q, q, causal=True, block_size=2)
print(np.asarray(out).shape)  # (2, 3, 5, 4)
```

## Performance summary

MacBook Air (Apple M4, 10-core CPU + 10-core GPU, 16 GB unified memory, fanless), B=1, H=8, D=64, non-causal:

| Sequence | CPU naive 1T | CPU flash 1T | CPU 10T | **GPU flash** | GPU naive |
|---:|---:|---:|---:|---:|---:|
| 128 | 5.81 ms | 3.75 ms | 0.660 ms | **0.538 ms** | 2.19 ms |
| 1024 | 458 ms | 227 ms | 34.7 ms | **3.15 ms** | 33.1 ms |
| 4096 | 7206 ms | 3570 ms | 576 ms | **47.83 ms** | 537.5 ms |

- Max absolute error vs CPU naive: **≤ 4.2e-6** (test tolerance `1e-4`).
- At S=4096 the GPU is **74.7×** faster than single-thread CPU, **12.1×** faster than 10-thread CPU, and **11.2×** faster than GPU naive.
- At S=128 the GPU is only **1.2×** faster than 10-thread CPU: startup/launch dominates, so CPU remains competitive for small one-off calls.
- GPU flash throughput is **0.72–0.74 TFLOP/s**, limited by SIMD reductions, exp, and shared-memory traffic; no absolute peak-efficiency claim is made because a simple FMA microbenchmark is easily optimized away by the Metal compiler.
- GPU numbers above are from the **buffer-resident** benchmark. The convenience API `flash_attention` allocates Q/K/V/O buffers per call; measured single calls are 2–3× higher (about 3–4.5 ms at S=512, 6.5–12 ms at S=1024).

See [GPU_COMPARISON.md](GPU_COMPARISON.md) (Chinese) or [GPU_COMPARISON_EN.md](GPU_COMPARISON_EN.md) for the full analysis, charts, memory comparison, tile tuning, and startup/transfer costs.

## Documentation

- [GPU_COMPARISON.md](GPU_COMPARISON.md): full CPU vs GPU comparison (Chinese)
- [GPU_COMPARISON_EN.md](GPU_COMPARISON_EN.md): English summary
- [ARCHITECTURE.md](ARCHITECTURE.md): layering, data flow, module responsibilities
- [results/tables.md](results/tables.md): generated full data tables

## License and acknowledgements

Learning project. The algorithm follows the FlashAttention paper (Dao et al., 2022) and the standard online-softmax derivation.
