# 简历项目（CV）：Rust Flash Attention — CPU / Apple Metal GPU 双后端与性能分析

> 本文把项目的 README、架构文档和性能报告压缩成 **可直接用于简历 / 面试的项目材料**（CV = Curriculum Vitae）。
> 个人项目 · 2026.09 – 2026.11 · 角色：独立开发 · 代码：Rust + Metal Shading Language + Python。
> 说明：这是一个**学习型项目，不是生产库**；面试时应主动说明边界，见第 9 节。

---

## 0. 一页纸摘要

**做了什么**：从零实现一个 Flash Attention / SDPA 库，包含纯 Rust CPU 后端与 macOS Metal GPU 后端；同时实现朴素 attention 作为正确性参照，并构建 6 个实现的对比实验。

**核心结果**（Apple M4，B=1、H=8、D=64、S=4096，非 causal）：

- GPU Flash **47.8 ms**，CPU 原版单线程 **3 570 ms**（**74.7×**），CPU 优化 10 线程 **576 ms**（**12.1×**）。
- 相对“物化 S×S”的 GPU naive：**11.2×** 更快；显存 **32 MiB vs 512 MiB（16×）**。
- 全部实现相对 CPU 朴素实现最大误差 **≤ 4.2×10⁻⁶**；13 个测试通过。
- GPU kernel-only 吞吐 **0.72–0.74 TFLOP/s**；未使用 simdgroup MMA，仍有优化空间。
- CPU 优化版：单线程 **≈1.4×**，10 线程 **4.45×**（44% 并行效率）。

**关键词**：Flash Attention、online softmax、tiling、Metal / MSL、subwarp / SIMD 归约、GPU kernel、Rust、PyO3、性能基准、roofline 分析。

---

## 1. 简历版（可直接粘贴）

### 项目条目

**Rust Flash Attention：CPU / Apple Metal GPU 双后端与性能分析** | 个人项目 | 2026.09 – 2026.11

技术栈：Rust (edition 2024)、Apple Metal / MSL、`std::thread`、PyO3、Python / matplotlib。

### 中文 bullet（建议 4 条）

- 独立实现 CPU + Metal GPU 双后端 Flash Attention，支持多 batch/head、causal mask、自定义 scale/block；13 个测试覆盖 D=8–128、奇数序列、causal 与大 scale，相对朴素实现最大误差 **4.2×10⁻⁶**。
- 设计 Metal kernel：**4/8 lane per query、Q/O 寄存器常驻、32 KiB threadgroup K/V tile、8-key 子块在线 softmax**；在 Apple M4 上 S=4096 达 **47.8 ms**，相对单线程 CPU **3.57 s** 提速 **74.7×**、相对 10 线程 CPU **576 ms** 提速 **12.1×**。
- 优化 CPU 路径：缓存 QK tile、复用 scratch、按行 `std::thread` 并行；单线程 **≈1.4×**、10 线程 **4.45×**，为 GPU 对比提供“非稻草人”的优化基线。
- 构建 6 实现 × 多形状 × causal/线程/tile 的 benchmark，输出 **9 张图表 + 技术报告**；量化 Flash Attention 相对物化 S×S 的 **16× 显存优势**（32 MiB vs 512 MiB @ S=4096）与 GPU naive 的显存墙。

### English bullets (4 bullets)

- Built a from-scratch Flash Attention library in Rust with CPU and Apple Metal GPU backends, supporting multi-batch/head inputs, causal masking, and configurable scale/block sizes; 13 tests cover D=8–128, odd sequence lengths, causal masks, and large scales, with **≤4.2e-6** max error against a naive f32 reference.
- Designed a Metal compute kernel with **4/8 lanes per query, register-resident Q/O, 32 KiB threadgroup K/V tiles, and 8-key sub-tile online softmax**; achieved **47.8 ms** at S=4096 on Apple M4, **74.7×** faster than a single-threaded CPU (**3.57 s**) and **12.1×** faster than a 10-thread CPU (**576 ms**).
- Optimized the CPU path with cached QK tiles, scratch reuse, and row-level `std::thread` parallelism (**≈1.4×** single-thread, **4.45×** at 10 threads), providing an optimized—not strawman—baseline for the GPU comparison.
- Built a reproducible 6-implementation benchmark across shapes, causal masking, CPU thread counts, and tile sizes; produced **9 charts and a technical report**, quantifying a **16× memory reduction** vs materialized S×S attention (32 MiB vs 512 MiB at S=4096).

### 一行版（简历空间很紧时）

> Rust / Metal Flash Attention：CPU + GPU 双后端，S=4096 相对 10 线程 CPU 提速 12.1×，显存较 S×S 物化方案降低 16×，误差 ≤4.2e-6。

---

## 2. 项目背景与目标

Transformer 的 scaled dot-product attention 需要计算 `S = QKᵀ`、`P = softmax(S)`、`O = PV`。直接实现会物化 `S×S` 矩阵，显存为 **O(S²)**；序列变长时成为瓶颈。

Flash Attention 的核心思路是把 K/V 分块，用 **online softmax** 维护 `m`（当前最大 logit）、`l`（分母）、`o`（输出累积），从而不物化完整的 `S×S` 矩阵，显存降为 **O(S·D)**。

项目目标：

1. 用纯 Rust 实现一个正确、可读的 CPU 版（零核心依赖），作为算法基准。
2. 用 Apple Metal 实现 GPU 版，理解 GPU 的线程映射、threadgroup memory、SIMD 归约与在线 softmax 的并行化。
3. 不只报“GPU 比 CPU 快”，而是从**正确性、延迟、吞吐、扩展性、causal、tile 调参、显存、启动/传输、可移植性**多个维度做可复现对比。
4. 保留 naive 实现与优化 CPU 版，避免“拿 GPU 对比未优化 CPU”的失真结论。

---

## 3. 我做了什么

### 3.1 CPU 后端（`src/attention.rs` + `src/cpu_opt.rs`）

- `Tensor4`：`[B, H, S, D]` 连续布局，统一索引与构造。
- `naive_attention` / `naive_attention_causal`：逐 query 全量 softmax，作为正确性参照。
- `scaled_dot_product_attention`：分块 + 在线 softmax，支持 causal、自定义 scale/block。
- 优化版：
  - 一次 QK 写入 `scores[block]`，避免“求 max / 求 exp”重复计算；
  - 复用 `o` 与 `scores` scratch，避免内层循环分配；
  - 按 `B·H·S` 行切块，`std::thread::scope` 并行，无锁、无 unsafe。
- 结果：单线程 ≈1.4×，10 线程 4.45×（44% 并行效率，受 4P+6E 与内存带宽限制）。

### 3.2 GPU 后端（`src/gpu/mod.rs` + `src/gpu/attention.metal`）

- **线程映射**：每 4 个 lane（D≤64）或 8 个 lane（D=128）负责一个 query；每个 lane 持有 `q[dim/lanes]`、`o[dim/lanes]` 寄存器。
- **K/V tiling**：threadgroup 协作搬入 `[bc][dim]` K/V tile（float4），大小受 32 KiB threadgroup 限制；一个 tile 被 `br` 个 query 复用。
- **在线 softmax 子块化**：每 8 个 key 计算 8 个 score 存入寄存器，求块最大，一次 rescale，再对 8 个 exp 独立累加 P·V；消除跨 key 的 `exp → m/l/o` 串行依赖。
- **SIMD 归约**：`simd_shuffle_xor` 做 4/8 lane 点积归约；causal mask 用 `score = -inf` 而不是分支，避免 SIMD-group 发散。
- **工程化**：按 head dim 运行时特化 MSL（`LANE_DIM`/`QUERY_LANES`）、pipeline 缓存、`GpuError` 不 panic、buffer 复用 benchmark。
- **正确性**：GPU flash 与 GPU naive 都与 CPU naive 逐元素对比；覆盖 D=8/16/32/64/128、S=777/333/1000、causal、scale=8、尾块。
- **naive 基线**：`scores → softmax → P·V` 三个 kernel，物化 S×S，用于显存墙对照。

### 3.3 对比与工程化

- 6 个实现：CPU naive、CPU flash 原版、CPU cached、CPU parallel、GPU flash、GPU naive。
- 指标：per-call 延迟、kernel-only 吞吐、end-to-end（含传输）、TFLOPS、max/mean/relative 误差、GPU 显存、线程/tile 参数。
- 协议：GPU 同时报告 per-call / amortized / e2e，避免把启动开销算进 kernel 吞吐；CPU 与 GPU 交替测量；无风扇机器上取短突发中位数并注明噪声。
- 产物：`results/*.csv`、`results/meta.txt`、`results/tables.md`、9 张图表、`GPU_COMPARISON.md`。
- 工程：`metal`/`python` feature-gated；CPU 核心零依赖；MSL 运行时编译，无 build.rs；`cargo test` 8 个、`cargo test --features metal` 13 个测试。

---

## 4. 核心量化结果

环境：MacBook Air（Mac16,12），Apple M4（10 核 CPU = 4P+6E，10 核 GPU），16 GB 统一内存，macOS 26.6.2，Xcode 27.2，Rust 1.99，Metal 4。B=1、H=8、D=64、非 causal。

| 维度 | 结果 |
|---|---|
| 正确性 | 13 个测试；相对 CPU naive 最大绝对误差 **4.2×10⁻⁶** |
| S=4096 延迟 | CPU 原版 1T **3 570 ms**；CPU 优化 10T **576 ms**；GPU **47.8 ms** |
| 相对加速 | GPU / CPU 1T **74.7×**；GPU / CPU 10T **12.1×**；GPU / GPU naive **11.2×** |
| S=1024 延迟 | CPU 1T 226.6 ms；CPU 10T 34.7 ms；GPU **3.15 ms**（11.0× vs 10T） |
| 显存（S=4096） | naive 分数矩阵 **512 MiB**；Flash Q/K/V/O **32 MiB**（**16×**） |
| Causal | GPU **1.6–2.0×**；CPU ≈2.0× |
| CPU 多线程 | 10 线程 **4.45×**（44% 效率） |
| GPU 吞吐 | **0.72–0.74 TFLOP/s**（kernel-only） |
| 启动/传输 | 设备/队列 ≈30 ms；MSL 编译 40–72 ms/head dim；热调用 0.78 ms；host↔device 19–76 GB/s |

> GPU 数字是 buffer-resident 的短突发中位数；便捷 API 每次分配 Q/K/V/O 会再高 2–3×。详见 `GPU_COMPARISON.md`。

---

## 5. 关键难点与解决（STAR）

### 难点 1：在线 softmax 的串行依赖

- **S**：最初每来一个 key 就更新 `m/l/o`，形成 `exp → new_m → l → o` 的跨 key 依赖链，GPU 的 exp 延迟无法被掩盖。
- **T**：在不增加 shared memory 的前提下打断依赖。
- **A**：每 8 个 key 为一个子块，8 个 score 先存入寄存器 `sreg[8]`，求块最大；每个子块只 rescale 一次 `o/l`，8 个 exp 相互独立，可被 SFU/流水线重叠。
- **R**：在线 softmax 语义不变，causal/非 causal 均通过测试；kernel-only 稳定在 0.72–0.74 TFLOP/s。

### 难点 2：32 KiB threadgroup 与 D=64/128 的取舍

- **S**：`[bc][D]` 的 K/V tile 要同时支持 D=64 与 D=128，且不能超过 `maxThreadgroupMemoryLength = 32 KiB`。
- **T**：在 tile 大小、每 query lane 数、每 lane 通道数之间取舍。
- **A**：D≤64 用 4 lane/query（每 lane 16 通道，2 次 shuffle 归约）；D=128 用 8 lane/query（每 lane 16 通道，3 次 shuffle）；`bc` 由 `32 KiB / (2·D·4)` 推导（D=64→64，D=128→32），并在 benchmark 中扫 `(bc, br)`。
- **R**：S=1024、D=64 最佳 **bc=32、br=32**（2.83 ms）；D=128 最佳 6.07 ms；避免 br=16 + bc=64 的 8.49 ms 退化配置。

### 难点 3：causal mask 与 SIMD 发散

- **S**：`simd_shuffle_xor` 要求 warp 内所有 lane 参与；在 key 循环中 `continue` 会让部分 lane 跳过 SIMD 操作，语义不安全。
- **T**：保持所有 lane 执行相同控制流，同时正确屏蔽未来 token。
- **A**：把 `j > i` 的 score 置为 `-INFINITY`，`exp(-inf)=0`，在线 softmax 自然忽略；`key_limit = min(S, (block+1)·br)` 只跳过整个 tile，不做 lane 级发散。
- **R**：causal 正确性通过测试；GPU causal 加速 **1.6–2.0×**，CPU ≈2.0×；解释 GPU 略低于 2× 的 tile 粒度原因。

### 难点 4：公平的 CPU/GPU 对比

- **S**：GPU 很容易“赢”一个未优化的 CPU 基线，得出夸大结论。
- **T**：让 CPU 也享受算法优化与多线程，区分“算法收益”和“硬件收益”。
- **A**：实现 cached（去重复 QK、复用 scratch）与 10 线程并行版；GPU 同时报告 per-call、kernel-only、e2e；用同一 naive 参考计算误差。
- **R**：即使对比优化后的 10 线程 CPU，GPU 在 S=4096 仍有 **12.1×**；S=128 仅 1.2×，诚实呈现小任务劣势。

### 难点 5：无风扇机器的计时噪声

- **S**：MacBook Air 无风扇，GPU 持续负载降频；per-call 抖动可达 ±40%。
- **T**：让结论不依赖单次计时。
- **A**：warm-up、短突发中位数、同时记录 kernel-only 与 per-call、重复运行；在报告中明确噪声区间。
- **R**：kernel-only 稳定在 ±5%，per-call 的保守口径不改变数量级结论。

---

## 6. 面试可展开的技术点

**Q1：Flash Attention 为什么省显存？**
A：标准 attention 物化 `S×S` 的 score/probability 矩阵，显存 O(S²)；Flash Attention 把 K/V 分块，用 online softmax 维护 `m/l/o`，每块只保留 `[bc][D]` 的 K/V 与 O(D) 的状态，显存 O(S·D)。代价是重算/额外的 rescale，用计算换显存。

**Q2：为什么用 8-key 子块，而不是逐 key 更新？**
A：逐 key 更新会产生 `exp → m/l/o` 的循环依赖，exp 延迟无法隐藏；8-key 子块把 8 个 exp 变成独立操作，同时只增加 8 个寄存器，不增加 shared memory。子块太大则寄存器压力和 rescale 开销上升。

**Q3：为什么 D=64 用 4 lane、D=128 用 8 lane？**
A：lane 越少，点积归约的 shuffle 次数越少（4 lane 2 次，8 lane 3 次），但每 lane 通道数/寄存器越多。D=64 用 4 lane 可把每 lane 控制在 16 通道；D=128 用 4 lane 会到 32 通道、寄存器压力过大，因此用 8 lane。

**Q4：为什么 GPU 只有 0.72–0.74 TFLOP/s？**
A：没有使用 simdgroup matrix（8×8 MMA），QK/PV 都是标量 FMA；每个 key 还要做 2–3 次 shuffle 归约、1 次 exp、shared memory 读写。瓶颈是指令/归约/特殊函数，而不是显存带宽。若用 `simdgroup_float8x8`、FP16/BF16、double buffering，预计还有 1.5–3× 空间（未实现）。

**Q5：causal 为什么 GPU 只有 1.6–2×，而 CPU 接近 2×？**
A：CPU 逐 query 精确跳过 `j > i` 的 key；GPU 以 `br` 个 query 为一个 threadgroup，块内低编号 query 会为高编号 query 的 key 做一部分“被 mask 的无效计算”，tile 越粗浪费越大。

**Q6：CPU 10 线程为什么只有 4.45×？**
A：4 个性能核 + 6 个能效核；每行独立但共享内存带宽；每次调用新建线程有开销。4 线程内接近线性（2.97×），之后效率下降。

**Q7：如何保证正确性？**
A：CPU naive 作为参考；CPU flash/cached/parallel、GPU flash/naive 全部逐元素对比；覆盖 D=8–128、奇数序列、causal、scale=8、尾块；13 个测试；benchmark 中记录 max/mean/relative 误差。

**Q8：为什么选 Metal？**
A：测试机是 Apple M4 MacBook Air，macOS 上没有 CUDA；Metal 是唯一可用的通用 GPU 计算接口。MSL 在运行时编译，无需 build.rs。若要支持 NVIDIA，需要另写 CUDA/HIP 后端。

**Q9：如果继续优化，优先做什么？**
A：① `simdgroup_float8x8` 重写 QK/PV；② FP16/BF16 + f32 softmax 累积；③ double buffering / async copy；④ 减少 shuffle 与 exp；⑤ CUDA 后端；⑥ 与 PyTorch SDPA / MPSGraph 对比；⑦ f64 参考与能耗测量。

**Q10：这个项目最大的收获？**
A：把“算法—GPU 执行模型—benchmark 方法论”连起来：同一个 online softmax，在 CPU 上是缓存/线程问题，在 GPU 上是 lane 映射、shared memory、SIMD 归约与 exp 延迟问题；而不夸大的对比需要优化 CPU 基线、区分 per-call/kernel/e2e，并诚实标注机器噪声。

---

## 7. 适用岗位与关键词

**适用**：推理/训练框架、GPU 算子、CUDA/Metal 性能优化、系统软件、高性能计算、Rust 开发。

**关键词**：Flash Attention、online softmax、tiling、SIMD / subwarp reduction、Metal、MSL、GPU kernel、Rust、PyO3、benchmark、roofline、performance analysis。

---

## 8. 项目结构与文档索引

| 文档 / 目录 | 内容 |
|---|---|
| `README_CN.md` / `README.md` | 项目介绍、构建、API、性能摘要、软硬件环境 |
| `ARCHITECTURE.md` | 分层架构、CPU/GPU 数据流、模块职责、限制 |
| `GPU_COMPARISON.md` | CPU vs GPU 全方位对比（中文，9 张图表 + 数据表） |
| `GPU_COMPARISON_EN.md` | 英文摘要 |
| `results/tables.md` | 自动生成的完整数据表（含环境、启动、误差、显存） |
| `results/charts/` | 延迟、吞吐、causal、线程、tile、传输、显存、误差图表 |
| `src/attention.rs` / `src/cpu_opt.rs` | CPU 核心与优化版 |
| `src/gpu/mod.rs` / `src/gpu/attention.metal` | Metal 驱动与 4 个 MSL kernel |
| `src/bin/compare.rs` / `scripts/report.py` | 对比 harness 与图表生成 |

代码量：GPU 后端约 **1.4k 行**（Rust + MSL），CPU 优化约 **400 行**，benchmark/report 约 **1.9k 行**，总计约 **4.7k 行**（含测试与脚本）。

---

## 9. 诚实说明（面试时主动说）

- **性质**：个人学习项目，非生产库；没有与 PyTorch SDPA、MPSGraph、cuDNN 做对比。
- **平台**：仅 macOS / Metal；未实现 CUDA/ROCm；Python 绑定目前只走 CPU。
- **GPU 绝对性能**：0.72–0.74 TFLOP/s 是手写标量 kernel 的水平，未用 simdgroup MMA；不要把它包装成“接近硬件峰值”。
- **形状限制**：head dim 必须是 8 的倍数且 ≤128；block 必须是 16 的倍数。
- **测量**：无风扇 MacBook Air，per-call 抖动可达 ±40%；GPU 数据是短突发中位数，换机器/散热会变。
- **数值**：只与 f32 CPU naive 对比，没有 f64 参考；误差 4.2e-6 是“与 f32 参考的差异”，不是绝对误差上界。
- **未做**：能耗/频率测量、CI 矩阵、GPU Python 绑定、多 GPU/分布式。

---

## 10. 复现

```bash
# CPU 测试（8 个）
cargo test

# CPU + GPU 测试（13 个）
cargo test --features metal

# CPU/GPU demo
cargo run --release --features metal

# 全量对比实验（约 2–3 分钟）
cargo run --release --features metal --bin compare

# 图表 + 表格
MPLCONFIGDIR=$PWD/.matplotlib-cache .venv/bin/python scripts/report.py
```
