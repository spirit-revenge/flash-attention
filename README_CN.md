# Flash Attention（Rust：CPU + Metal GPU）

中文 | [English](README.md)

一个用 Rust 实现的 **Flash Attention / Scaled Dot-Product Attention** 学习型项目，包含两条后端：

- **CPU 版**：纯 Rust、核心零依赖；提供朴素实现、原始分块实现、缓存优化版与多线程版；可选 PyO3/Python 绑定。
- **GPU 版**：macOS **Metal** 实现（MSL compute kernel + Rust 驱动），按 query/lane 映射、K/V tile 复用、8-key 子块在线 softmax；同时提供物化 S×S 的 naive 基线。

并附带一套可复现的 **CPU vs GPU 对比实验**（延迟、吞吐、误差、显存、线程扩展、tile 调参、传输/启动开销）。

> [!WARNING]
> 这是一个个人学习项目，请勿用于正式生产项目。

## 项目特性

- 纯 Rust CPU 实现，核心零依赖；分块 + 在线 softmax，不物化 S×S 注意力矩阵
- 多 batch / 多 head、causal mask、自定义 scale 与 block size
- CPU 优化版：tile 分数缓存、scratch 复用、行级多线程（`std::thread`，无外部依赖）
- Metal GPU Flash Attention：4/8 lane per query、寄存器 Q/O、32 KiB threadgroup、MSL 按 head dim 特化
- Metal GPU naive 基线：`scores → softmax → P·V` 三 kernel，用于量化 O(S²) 显存代价
- 可选 PyO3 Python 扩展（CPU 路径）
- 对比 harness + 图表/表格生成脚本

## 目录结构

```
flash-attention/
├── Cargo.toml
├── README.md / README_CN.md
├── GPU_COMPARISON.md          # 中文：CPU vs GPU 全方位对比（含图表）
├── GPU_COMPARISON_EN.md       # English summary
├── ARCHITECTURE.md            # 架构说明
├── src/
│   ├── attention.rs           # CPU 核心：Tensor4、naive、分块在线 softmax
│   ├── cpu_opt.rs             # CPU 缓存版 + 多线程版
│   ├── gpu/
│   │   ├── mod.rs             # Metal 设备/管线/缓冲/调度/计时
│   │   └── attention.metal    # 4 个 MSL kernel
│   ├── lib.rs                 # 导出 + 可选 Python 绑定
│   ├── main.rs                # CPU/GPU demo
│   └── bin/compare.rs         # 对比实验 harness
├── scripts/report.py          # CSV → 图表 + Markdown 表格
├── results/                   # 原始数据、图表、表格（已生成）
└── test.py                    # Python smoke test
```

## 快速开始

### 1. CPU：构建与测试

```bash
cargo build --release
cargo test
```

运行 CPU/GPU demo（GPU 需要 `--features metal`）：

```bash
cargo run --release --features metal
```

### 2. GPU（macOS + Metal）

要求：macOS（Metal 2.1+；本项目在 macOS 26 / Apple M4 上验证），Rust 1.85+。MSL 在运行时编译，无需 `build.rs`，但需要可用的 Metal 运行时。

```bash
cargo test --features metal
cargo run --release --features metal   # CPU/GPU demo
```

Rust API 示例：

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
            block_size: 32,   // 必须是 16 的倍数
            query_block: 32,  // query_block * lanes % 32 == 0
            ..Default::default()
        },
    )
    .expect("attention");

println!("shape: {:?}", (out.batch, out.heads, out.seq, out.dim));
```

GPU 路径的限制：

- head dim 必须是 8 的倍数且 ≤ 128（64/128 为典型值）；
- `block_size` 必须是 16 的倍数，且不超过 `max_block_size(dim)`（D=64 为 64，D=128 为 32）；
- `query_block * lanes` 必须是 32 的倍数（D≤64 时 lanes=4，D=128 时 lanes=8）；
- 仅 macOS；Python 绑定目前只走 CPU。

### 3. CPU vs GPU 对比实验

```bash
# 全量实验（约 2–3 分钟）
cargo run --release --features metal --bin compare

# 快速冒烟
cargo run --release --features metal --bin compare -- --quick

# 生成图表 + Markdown 表格
python3 -m venv .venv
.venv/bin/pip install pandas matplotlib
MPLCONFIGDIR=$PWD/.matplotlib-cache .venv/bin/python scripts/report.py
```

输出：`results/timings.csv`、`results/blocksize.csv`、`results/threads.csv`、`results/transfer.csv`、`results/startup.csv`、`results/meta.txt`、`results/tables.md`、`results/charts/*.png`。

### 4. Python（CPU）

```bash
python3 -m pip install numpy maturin
maturin build --release --features python
python3 -m pip install --force-reinstall target/wheels/*.whl
python3 test.py
```

输入形状支持 `[batch, heads, seq, dim]`，也支持 `[seq, dim]` 等低维输入：

```python
import numpy as np
import flash_attention

q = np.random.randn(2, 3, 5, 4).astype(np.float32)
out = flash_attention.flash_attention_py(q, q, q, causal=True, block_size=2)
print(np.asarray(out).shape)  # (2, 3, 5, 4)
```

## 性能摘要

测试环境：MacBook Air（Apple M4，10 核 CPU + 10 核 GPU，16 GB 统一内存，无风扇），B=1、H=8、D=64、非 causal：

| 序列长度 | CPU naive 1T | CPU flash 1T | CPU 10 线程 | **GPU flash** | GPU naive |
|---:|---:|---:|---:|---:|---:|
| 128 | 5.81 ms | 3.75 ms | 0.660 ms | **0.538 ms** | 2.19 ms |
| 1024 | 458 ms | 227 ms | 34.7 ms | **3.15 ms** | 33.1 ms |
| 4096 | 7206 ms | 3570 ms | 576 ms | **47.83 ms** | 537.5 ms |

- 最大绝对误差相对 CPU naive：**≤ 4.2×10⁻⁶**（测试阈值 `1e-4`）。
- S=4096 时 GPU 相对 CPU 单线程 **74.7×**、相对 10 线程 **12.1×**、相对 GPU naive **11.2×**。
- S=128 时 GPU 相对 10 线程仅 **1.2×**：启动/同步开销占比高，小任务仍建议 CPU。
- GPU flash 吞吐约 **0.72–0.74 TFLOP/s**，受 SIMD 归约、exp 与 shared memory 访问限制；不做绝对峰值效率的宣称（Metal 上的朴素 FMA 微基准很容易被编译器优化掉）。
- 表中 GPU 数字是 **复用 buffer 的 resident 基准**；便捷 API `flash_attention` 每次会分配 Q/K/V/O buffer，单次调用实测约再高 2–3×（S=512 约 3–4.5 ms，S=1024 约 6.5–12 ms）。

完整分析、图表、显存对比、tile 调参与启动/传输成本见 [GPU_COMPARISON.md](GPU_COMPARISON.md)。

## 文档

- [GPU_COMPARISON.md](GPU_COMPARISON.md)：CPU vs GPU 全方位对比（中文）
- [GPU_COMPARISON_EN.md](GPU_COMPARISON_EN.md)：English summary
- [ARCHITECTURE.md](ARCHITECTURE.md)：分层架构、数据流、模块职责
- [results/tables.md](results/tables.md)：自动生成的完整数据表

## 许可与致谢

本项目为学习用途，算法参考 FlashAttention 论文（Dao et al., 2022）与在线 softmax 的经典推导。请勿用于正式项目。
