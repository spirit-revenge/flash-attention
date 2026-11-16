# Flash Attention 项目架构

[中文](ARCHITECTURE.md) | English

> 本文描述项目的整体架构、分层设计、核心模块、GPU 后端与数据流。基准测试结果与横向对比见 [GPU_COMPARISON.md](GPU_COMPARISON.md)。

## 项目概览

**`flash-attention`** 是一个用 **Rust** 实现的 Flash Attention / SDPA（缩放点积注意力）学习型项目，包含两条后端：

- **CPU 后端**：纯 Rust、核心零依赖，通过分块 + 在线 softmax 避免物化 `S×S` 注意力矩阵；同时保留朴素实现作为正确性参照；可选 PyO3/Python 绑定。
- **GPU 后端**：macOS **Metal** 实现（MSL kernel + Rust 驱动），按 query/lane 映射、K/V tile 放入 threadgroup memory、子块在线 softmax；并提供一条“物化 S×S”的 naive 基线用于对比。

此外提供一套可复现的 CPU vs GPU 对比 harness（`src/bin/compare.rs`）和图表生成脚本（`scripts/report.py`）。

当前测试规模：`cargo test` 8 个 CPU 测试；`cargo test --features metal` 13 个测试（含 5 个 GPU 测试）全部通过。

## 技术栈

| 维度 | 选型 |
|------|------|
| 语言 | Rust (edition 2024)；GPU kernel 使用 Metal Shading Language |
| 构建 | Cargo，crate-type = `cdylib` + `rlib` |
| CPU 并行 | `std::thread::scope`（零外部依赖） |
| Python 绑定 | PyO3 0.22.6（可选 feature `python`） |
| GPU 绑定 | `metal` 0.33 + `objc` 0.2（可选 feature `metal`，仅 macOS） |
| Python 打包 | maturin |
| 核心依赖 | CPU 路径零依赖；GPU 路径仅 Metal 相关 crate |

## 目录结构

```
flash-attention/
├── Cargo.toml              # 包定义、python/metal feature、compare bin
├── Cargo.lock
├── README.md / README_CN.md
├── GPU_COMPARISON.md       # 中文对比报告（图表 + 表格 + 分析）
├── GPU_COMPARISON_EN.md    # 英文摘要
├── ARCHITECTURE.md         # 本文
├── test.py                 # Python 端 smoke test
├── src/
│   ├── attention.rs        # CPU 核心算法层（628 行）
│   ├── cpu_opt.rs          # CPU 缓存版 + 多线程版（406 行）
│   ├── gpu/
│   │   ├── mod.rs          # Metal 设备/管线/缓冲/调度/计时（1042 行）
│   │   └── attention.metal # 4 个 MSL kernel（329 行）
│   ├── lib.rs              # 库入口 + 可选 Python 绑定（269 行）
│   ├── main.rs             # CPU/GPU demo（77 行）
│   └── bin/compare.rs      # 对比实验 harness（1263 行）
├── scripts/report.py       # CSV → 图表 + Markdown 表格
├── results/                # 原始 CSV、meta、图表、tables.md
└── target/                 # 构建产物（gitignore）
```

## 分层架构

项目本质上是**四层结构**：

```
┌──────────────────────────────────────────────────────────────┐
│ 接入层 (Presentation)                                        │
│  ├─ src/main.rs        CPU/GPU demo                          │
│  ├─ src/bin/compare.rs 对比实验 harness                      │
│  ├─ test.py            Python smoke test                     │
│  └─ scripts/report.py  CSV → 图表/表格                       │
├──────────────────────────────────────────────────────────────┤
│ 绑定/适配层                                                  │
│  ├─ lib.rs           re-export + 可选 PyO3 模块              │
│  ├─ normalize_shape  1D~4D → (B,H,S,D)                       │
│  └─ nested_to_flat_and_shape / tensor4_to_python_nested      │
├──────────────────────────────────────────────────────────────┤
│ 核心算法层                                                   │
│  ├─ attention.rs   CPU：Tensor4 / naive / 分块在线 softmax   │
│  └─ cpu_opt.rs     CPU：缓存 tile / scratch 复用 / 多线程    │
├──────────────────────────────────────────────────────────────┤
│ GPU 后端                                                     │
│  ├─ gpu/mod.rs     Rust 驱动：device/queue/pipeline/buffer   │
│  └─ attention.metal MSL：flash / scores / softmax / pv       │
└──────────────────────────────────────────────────────────────┘
```

## CPU 核心算法层

### 1. `Tensor4`：数据模型与内存布局

- 固定 4 维逻辑形状 `[batch, heads, seq, dim]`，底层是连续 `Vec<f32>`。
- 索引公式：`((b * heads + h) * seq + s) * dim + d`（`Tensor4::index`）。
- 提供 `new/zeros/from_vec/from_fn/from_random` 多种构造方式；`from_vec` 强制校验长度一致。

### 2. 参考实现（naive）

`naive_attention` / `naive_attention_causal`：逐 query 计算完整 score 行，做 softmax，再对 V 加权求和。语义最直观，作为正确性参照；每行只分配 O(S) scratch，因此 CPU naive 的**内存其实很省**，只是计算慢。

### 3. 核心：分块 + 在线 softmax

`scaled_dot_product_attention` 是原版核心：

- 对每个 `(b,h,i)` 维护 `m`（当前最大 logit）、`l`（softmax 分母）、`o`（输出累积）；
- K/V 按 `block_size` 分块，块内先求 `block_m`，再 `exp(score - block_m)` 累积；
- 块间用 `alpha = exp(m - new_m)`、`beta = exp(block_m - new_m)` 合并；
- 支持自定义 `scale`、causal mask、任意 `block_size`；
- 不物化 `S×S` 矩阵。

已知低效点：每个 tile 的 QK 分数被计算两次（求 max 与求 exp 各一次）；内层循环反复分配 `o`/`block_o`；单线程。

### 4. CPU 优化版（`cpu_opt.rs`）

- `scaled_dot_product_attention_cached`：一次 QK 写入 `scores[block]`，复用 `o`/`scores`，4 路累加点积；
- `scaled_dot_product_attention_parallel` / `flash_attention_parallel`：把 `B*H*S` 行切成连续块，`std::thread::scope` 每块一个线程，无锁、无 unsafe；
- `naive_attention_parallel`：naive 的并行版，用于对照 CPU 线程扩展性。

优化版相对原版约 1.4×（单线程），10 线程相对单线程约 4.5×（M4 4P+6E）。它不是“极致优化”，但足以把算法优化与硬件提升分开。

## GPU 后端（Metal）

### 1. Rust 驱动（`gpu/mod.rs`）

- `GpuAttention::new()`：创建 `Device` / `CommandQueue`，读取 `GpuInfo`（名称、统一内存、32 KiB threadgroup、8.88 GiB max buffer 等）。
- `prepare()`：校验形状/选项、按 `dim` 编译或复用 pipeline、分配 shared-storage buffer、计算 grid/threadgroup 与 threadgroup memory 大小。
- `flash_attention` / `naive_attention`：便捷 API（分配 + 上传 + dispatch + 读回）。
- `benchmark_flash` / `benchmark_naive`：resident 基准，分别报告 per-call、amortized kernel-only、end-to-end。
- `transfer_benchmark`：host↔device、device↔device 带宽。
- `warmup_dim` / `compiled_dims` / `flash_pipeline_limits`：编译成本与 occupancy 诊断。
- 错误类型 `GpuError`：NoDevice / Shape / Options / Memory / Compile / Pipeline / Unsupported；不 panic。

GPU 约束：

- head dim ∈ {8, 16, …, 128}（必须是 8 的倍数且 ≤128）；
- `block_size` 必须是 16 的倍数且 ≤ `max_block_size(dim)`（由 32 KiB threadgroup 上限推导）；
- `query_block * lanes % 32 == 0`（lanes = 4 if D≤64 else 8）；
- 张量元素数 ≤ `u32::MAX`（kernel 用 `uint` 索引）。

### 2. MSL kernel（`attention.metal`）

**Flash Attention kernel**

- 一个 threadgroup 处理 `br` 个 query（`block_rows`）+ 一个 `bc` 个 key 的 K/V tile；
- 每 4 个 lane（D≤64）或 8 个 lane（D=128）负责一个 query，lane 负责连续的 `dim/lanes` 个通道；
- 每个 lane 在寄存器里保存 `q[dim/lanes]`、`o[dim/lanes]`；
- K/V tile 协作搬入 threadgroup memory（float4），两次 barrier；
- 每 8 个 key 为一个子块：计算 8 个 score 存入寄存器 `sreg[8]`，求块最大，一次 rescale，8 个 exp 独立计算并累加 P·V；
- 点积用 4 个独立累加器 + `simd_shuffle_xor` 做 4/8 lane 归约；
- causal mask：`score = -inf`，不使用分支，避免 SIMD-group 发散；
- 编译器按 `LANE_DIM`/`QUERY_LANES` 特化，内层循环展开为 float4 访存。

**Naive 三步 kernel**

- `scores_kernel`：每个 `(i,j)` 一个线程，写 `scores[b,h,i,j]`（causal 写 `-inf`）；
- `softmax_rows_kernel`：每行一个线程，三趟扫描；
- `pv_kernel`：每个 `(i,d)` 一个线程，累加 `P·V`；
- 用于正确性对照和显存墙演示，不追求性能。

### 3. 为什么这样映射

- **query 级并行**：天然把 `B*H*S` 展开，GPU 有足够多线程；
- **K/V tile 复用**：一个 tile 被 `br` 个 query 共享，降低 global 读放大；
- **q/o 寄存器**：每 lane 只访问自己的通道，避免 shared memory 往返；
- **子块化在线 softmax**：打断 `exp → m/l/o` 的跨 key 依赖链；
- **避免 SIMD 发散**：causal 用 `-inf` 而不是 `continue`。

## 数据流

```
CPU 路径:
  make_qkv_example / Python 输入
    → Tensor4{q,k,v}
    → scaled_dot_product_attention(_cached/_parallel)
    → Tensor4 out
    → main / benchmark / Python nested list

GPU 路径:
  Tensor4{q,k,v}
    → GpuAttention::prepare (校验 + pipeline + buffers)
    → upload (memcpy host→shared buffer)
    → flash_attention_kernel / naive 三步
    → waitUntilCompleted
    → download (shared buffer→Vec<f32>)
    → Tensor4 out

对比路径:
  Case (B,H,S,D,causal)
    → CPU naive reference
    → GPU flash / GPU naive (benchmark_*)
    → CPU flash / cached / parallel
    → Row {latency, throughput, e2e, tflops, error, gpu_bytes}
    → results/*.csv → scripts/report.py → charts + tables.md
```

## 正确性策略

1. **双实现互证**：CPU naive ↔ CPU flash ↔ CPU cached/parallel ↔ GPU flash ↔ GPU naive。
2. **单元测试**：CPU 8 个（基础、多 batch/head、causal、shape suite、缓存、并行、naive 并行）；GPU 5 个（flash vs naive、naive vs naive、长序列/大 scale、非法参数、buffer 往返）。
3. **全量 benchmark 误差统计**：每个形状记录 max abs / mean abs / relative L2；GPU 最大绝对误差约 4e-6。
4. **边界**：D=8..128、S=777/333/1000、block=16、causal、scale=8、尾块。
5. **失败可诊断**：GPU 错误返回 `GpuError`，包含形状/选项/内存/编译信息。

## 基准测试与可复现性

- `src/bin/compare.rs` 统一测六个实现，输出 `results/*.csv` + `meta.txt`；
- GPU 同时报告 per-call / amortized / e2e，避免把启动开销算进 kernel 吞吐；
- `results/tables.md` 与 `results/charts/*.png` 由 `scripts/report.py` 从 CSV 生成，报告中的数字可追溯；
- 机器元数据（GPU 名称、限制、rustc 版本）写入 `meta.txt`。

## 已知限制与可改进点

**CPU**

- 原版仍有重复 QK 与内层分配；优化版已修复，但未用 NEON intrinsics / 常驻线程池；
- 多线程版每次调用新建线程，短任务有开销；
- Python 绑定大张量走嵌套 list，序列化开销大。

**GPU**

- 仅 macOS/Metal；D 必须是 8 的倍数且 ≤128；block 必须是 16 的倍数；
- 未使用 simdgroup matrix（8×8 MMA），标量 FMA + shuffle 限制了吞吐；
- 未做 double buffering / async copy / FP16/BF16；
- 每个 head dim 需一次 40–72 ms MSL 编译；可缓存 binary archive；
- Python 绑定未暴露 GPU；
- 无 fanless 机器的持续频率/能耗测量。

**工程**

- 没有 CI 矩阵（Linux CPU-only、macOS GPU、Python wheel）；
- 没有 f64 参考实现来区分“与 f32 参考的差异”和“真实数值误差”。

## 后续演进建议

1. 用 `simdgroup_float8x8` 重写 QK/PV（预期 1.5–3×）；
2. FP16/BF16 + f32 softmax 累积；
3. double buffering + `async copy`；
4. 抽象设备层，增加 CUDA/ROCm 后端；
5. 暴露 GPU 的 Python binding；
6. CPU 常驻线程池 + NEON intrinsics；
7. 加入 CI 与能耗基准。
