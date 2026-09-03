use std::f32::consts::PI;
use std::time::Instant;

#[derive(Clone, Debug)]
/// `Tensor4` 表示一个 4 维张量，布局约定为：
/// `[batch, heads, seq, dim]`
///
/// 也就是说，连续存储中的一个元素位置可以被表示为：
/// `((b * heads + h) * seq + s) * dim + d`
///
/// 这个布局使得同一个 batch / head 下的所有 token 和维度能够按连续内存段访问，
/// 对注意力计算中的 Q、K、V 读取和输出写入都比较友好。
pub struct Tensor4 {
    /// batch 大小，表示样本数量。
    pub batch: usize,
    /// attention head 的数量。
    pub heads: usize,
    /// 序列长度，也就是每个样本中的 token 数量。
    pub seq: usize,
    /// 每个 token 的特征维度。
    pub dim: usize,
    /// 展平后的存储数据，长度为 `batch * heads * seq * dim`。
    pub data: Vec<f32>,
}

impl Tensor4 {
    /// 创建一个指定形状的全 0 张量。
    ///
    /// 这是最基础的构造方式，通常用于初始化输出张量，或者创建一个空白容器。
    pub fn new(batch: usize, heads: usize, seq: usize, dim: usize) -> Self {
        Self {
            batch,
            heads,
            seq,
            dim,
            data: vec![0.0; batch * heads * seq * dim],
        }
    }

    /// `zeros` 是 `new` 的语义别名，方便调用时读起来更自然。
    pub fn zeros(batch: usize, heads: usize, seq: usize, dim: usize) -> Self {
        Self::new(batch, heads, seq, dim)
    }

    /// 从已存在的 `Vec<f32>` 构造张量。
    ///
    /// 这里会强制校验长度是否和 `(batch * heads * seq * dim)` 完全一致，
    /// 这样可以尽早发现数据布局错误，避免后续对内存出现越界或错位访问。
    pub fn from_vec(batch: usize, heads: usize, seq: usize, dim: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), batch * heads * seq * dim);
        Self {
            batch,
            heads,
            seq,
            dim,
            data: values,
        }
    }

    /// 根据索引生成张量值。
    ///
    /// 这是一个非常方便的测试和数据构造方法：
    /// 传入一个闭包 `f(b, h, s, d)`，系统会按 `[b, h, s, d]` 的顺序填充每个元素。
    pub fn from_fn(
        batch: usize,
        heads: usize,
        seq: usize,
        dim: usize,
        mut f: impl FnMut(usize, usize, usize, usize) -> f32,
    ) -> Self {
        let mut data = Vec::with_capacity(batch * heads * seq * dim);
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    for d in 0..dim {
                        data.push(f(b, h, s, d));
                    }
                }
            }
        }
        Self::from_vec(batch, heads, seq, dim, data)
    }

    /// 生成满足一定随机分布的张量。
    ///
    /// 这里使用 `SplitMix64` 作为随机数源，然后通过 Box-Muller 变换生成近似标准正态分布，
    /// 这使得随机初始化更接近真实模型训练里的权重分布。
    pub fn from_random(batch: usize, heads: usize, seq: usize, dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let total = batch * heads * seq * dim;
        let mut data = Vec::with_capacity(total);

        for _ in 0..total {
            let x = rng.next_f32();
            let y = rng.next_f32();
            // Box-Muller 变换：把均匀随机数转换成近似高斯分布
            let z = (-2.0 * x.ln()).sqrt() * (2.0 * PI * y).cos();
            data.push(z * 0.5);
        }

        Self::from_vec(batch, heads, seq, dim, data)
    }

    /// 计算张量中的一个元素索引。
    ///
    /// 这是核心的内存布局函数：保证所有元素在 `Vec<f32>` 中的顺序和
    /// `(batch, heads, seq, dim)` 的逻辑索引完全一致。
    pub fn index(&self, b: usize, h: usize, s: usize, d: usize) -> usize {
        ((b * self.heads + h) * self.seq + s) * self.dim + d
    }
}

#[derive(Clone, Copy, Debug)]
/// `SplitMix64` 是一个简单而快速的伪随机数生成器，
/// 它常用于生成稳定可复现的随机初始化值。
///
/// 这里使用它来生成 Q/K/V 的随机值，自身不依赖任何外部库，
/// 因此适合对实现过程做可重复测试和 benchmark。
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// 生成下一条 64 位伪随机值。
    ///
    /// SplitMix64 的本质是一个线性混合器，做一些位运算和乘法混淆，
    /// 能在保持较低实现复杂度的同时提供不错的随机性。
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// 将随机值转换到 `[0, 1)` 的 f32 空间。
    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() as f64;
        (value / u64::MAX as f64) as f32
    }
}

/// 计算两个长度相同的向量点积。
///
/// 注意力中这里用来计算 query 与 key 的相似度：
/// `score = dot(q_i, k_j) / sqrt(dim)`
///
/// 这个函数通过固定 8 个元素一组的方式做分块计算，减少重复循环开销，
/// 在不牺牲清晰度的前提下提升了简化版 CPU 向量化效果。
fn dot_product_qk(q: &[f32], k: &[f32]) -> f32 {
    let chunks = q.len() / 8;
    let mut sum = 0.0;

    for i in 0..chunks {
        let base = i * 8;
        let q_chunk = &q[base..base + 8];
        let k_chunk = &k[base..base + 8];
        sum += q_chunk[0] * k_chunk[0]
            + q_chunk[1] * k_chunk[1]
            + q_chunk[2] * k_chunk[2]
            + q_chunk[3] * k_chunk[3]
            + q_chunk[4] * k_chunk[4]
            + q_chunk[5] * k_chunk[5]
            + q_chunk[6] * k_chunk[6]
            + q_chunk[7] * k_chunk[7];
    }

    let remainder_start = chunks * 8;
    for i in remainder_start..q.len() {
        sum += q[i] * k[i];
    }
    sum
}

/// 参考实现：按“逐 token 计算完整 softmax”的方式实现 attention。
///
/// 其逻辑是：
/// 1. 对每个 query `i`，先计算它和所有 key `j` 的 score；
/// 2. 对这些 score 做 softmax；
/// 3. 用 softmax 权重对 V 做加权求和，得到输出。
///
/// 这个版本的优点是语义最直观，能很好地作为“正确答案”对照，实现清晰但效率较低。
pub fn naive_attention(q: &Tensor4, k: &Tensor4, v: &Tensor4) -> Tensor4 {
    assert_eq!(q.batch, k.batch);
    assert_eq!(q.heads, k.heads);
    assert_eq!(q.batch, v.batch);
    assert_eq!(q.heads, v.heads);
    assert_eq!(q.dim, k.dim);
    assert_eq!(q.dim, v.dim);

    let mut out = Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    let scale = 1.0 / (q.dim as f32).sqrt();

    for b in 0..q.batch {
        for h in 0..q.heads {
            for i in 0..q.seq {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0; k.seq];

                for j in 0..k.seq {
                    let q_base = ((b * q.heads + h) * q.seq + i) * q.dim;
                    let k_base = ((b * k.heads + h) * k.seq + j) * k.dim;
                    let score = dot_product_qk(&q.data[q_base..q_base + q.dim], &k.data[k_base..k_base + k.dim]) * scale;
                    scores[j] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut exp_sum = 0.0;
                let mut weights = vec![0.0; k.seq];
                for j in 0..k.seq {
                    let w = (scores[j] - max_score).exp();
                    weights[j] = w;
                    exp_sum += w;
                }

                for d in 0..q.dim {
                    let mut acc = 0.0;
                    for j in 0..k.seq {
                        let v_idx = v.index(b, h, j, d);
                        acc += weights[j] * v.data[v_idx];
                    }
                    let idx = out.index(b, h, i, d);
                    out.data[idx] = acc / exp_sum;
                }
            }
        }
    }

    out
}

/// 参考实现：带 causal mask 的 attention。
///
/// 这里要求第 `i` 个位置只能 attend 到 `0..=i` 的历史 token，
/// 这样就能模拟 decoder-only 的自回归注意力。
///
/// 它和 `naive_attention` 的核心思想一致，只是 `j` 的范围被限制为 `0..=i`。
pub fn naive_attention_causal(q: &Tensor4, k: &Tensor4, v: &Tensor4) -> Tensor4 {
    assert_eq!(q.batch, k.batch);
    assert_eq!(q.heads, k.heads);
    assert_eq!(q.batch, v.batch);
    assert_eq!(q.heads, v.heads);
    assert_eq!(q.dim, k.dim);
    assert_eq!(q.dim, v.dim);

    let mut out = Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    let scale = 1.0 / (q.dim as f32).sqrt();

    for b in 0..q.batch {
        for h in 0..q.heads {
            for i in 0..q.seq {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0; i + 1];

                for j in 0..=i {
                    let q_base = ((b * q.heads + h) * q.seq + i) * q.dim;
                    let k_base = ((b * k.heads + h) * k.seq + j) * k.dim;
                    let score = dot_product_qk(&q.data[q_base..q_base + q.dim], &k.data[k_base..k_base + k.dim]) * scale;
                    scores[j] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut exp_sum = 0.0;
                let mut weights = vec![0.0; i + 1];
                for j in 0..=i {
                    let w = (scores[j] - max_score).exp();
                    weights[j] = w;
                    exp_sum += w;
                }

                for d in 0..q.dim {
                    let mut acc = 0.0;
                    for j in 0..=i {
                        let v_idx = v.index(b, h, j, d);
                        acc += weights[j] * v.data[v_idx];
                    }
                    let idx = out.index(b, h, i, d);
                    out.data[idx] = acc / exp_sum;
                }
            }
        }
    }

    out
}

/// 这是项目的核心 API：Scaled Dot-Product Attention (SDPA)。
///
/// 它在 `naive_attention` 的基础上做了两个关键优化：
/// 1. 使用 `scale` 参数控制缩放；
/// 2. 按 `block_size` 对 key/value 做分块处理，减少一次性构造大矩阵的成本。
///
/// 这是一种“在线 softmax”风格实现：
/// 不是先把整个 attention 矩阵算出来再做 softmax，
/// 而是在逐块处理时维护 `m`、`l` 和 `o`，最后得到稳定且正确的输出。
pub fn scaled_dot_product_attention(
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
    scale: Option<f32>,
    causal: bool,
    block_size: usize,
) -> Tensor4 {
    assert_eq!(q.batch, k.batch);
    assert_eq!(q.heads, k.heads);
    assert_eq!(q.batch, v.batch);
    assert_eq!(q.heads, v.heads);
    assert_eq!(q.dim, k.dim);
    assert_eq!(q.dim, v.dim);
    assert!(block_size > 0, "block_size must be greater than zero");

    let dim = q.dim;
    let scale = scale.unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
    let mut out = Tensor4::new(q.batch, q.heads, q.seq, q.dim);

    for b in 0..q.batch {
        for h in 0..q.heads {
            for i in 0..q.seq {
                // 在线 softmax 的状态：
                // m = 当前最大 logit
                // l = 当前 softmax 分母
                // o = 当前输出累积向量
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0;
                let mut o = vec![0.0; dim];

                let max_j = if causal { i + 1 } else { k.seq };

                for block_start in (0..max_j).step_by(block_size) {
                    let block_end = (block_start + block_size).min(max_j);
                    let mut block_m = f32::NEG_INFINITY;
                    let mut block_l = 0.0;
                    let mut block_o = vec![0.0; dim];

                    // 第一步：先计算当前 block 内的最大 score，便于数值稳定。
                    for j in block_start..block_end {
                        let q_base = ((b * q.heads + h) * q.seq + i) * q.dim;
                        let k_base = ((b * k.heads + h) * k.seq + j) * k.dim;
                        let score = dot_product_qk(&q.data[q_base..q_base + q.dim], &k.data[k_base..k_base + k.dim]) * scale;
                        if score > block_m {
                            block_m = score;
                        }
                    }

                    // 第二步：在稳定化后对每个 score 做 `exp(score - block_m)`，
                    // 再把对应的 V 加权累加到 `block_o` 中。
                    for j in block_start..block_end {
                        let q_base = ((b * q.heads + h) * q.seq + i) * q.dim;
                        let k_base = ((b * k.heads + h) * k.seq + j) * k.dim;
                        let score = dot_product_qk(&q.data[q_base..q_base + q.dim], &k.data[k_base..k_base + k.dim]) * scale;
                        let exp_score = (score - block_m).exp();
                        block_l += exp_score;

                        for d in 0..dim {
                            let v_idx = v.index(b, h, j, d);
                            block_o[d] += exp_score * v.data[v_idx];
                        }
                    }

                    // 使用在线 softmax 的合并规则：
                    // 如果已有历史状态 m 和 l，向当前 block 合并时需要考虑 rescaling。
                    let new_m = m.max(block_m);
                    let alpha = if m == f32::NEG_INFINITY { 0.0 } else { (m - new_m).exp() };
                    let beta = (block_m - new_m).exp();

                    for d in 0..dim {
                        o[d] = alpha * o[d] + beta * block_o[d];
                    }
                    l = alpha * l + beta * block_l;
                    m = new_m;
                }

                for d in 0..dim {
                    let idx = out.index(b, h, i, d);
                    out.data[idx] = o[d] / l;
                }
            }
        }
    }

    out
}

/// 提供一个更简洁的 API 入口：
/// 这是对 `scaled_dot_product_attention` 的包装，默认表示非 causal 的标准注意力。
pub fn flash_attention(q: &Tensor4, k: &Tensor4, v: &Tensor4, block_size: usize) -> Tensor4 {
    scaled_dot_product_attention(q, k, v, None, false, block_size)
}

/// CPU 版本的 flash_attention 包装函数。
///
/// 这个名字是为了更贴近“CPU 端高性能注意力”的语义，用于 Rust 侧和 demo 入口中调用。
pub fn flash_attention_cpu(q: &Tensor4, k: &Tensor4, v: &Tensor4, block_size: usize) -> Tensor4 {
    flash_attention(q, k, v, block_size)
}

/// 生成一组更真实、更有结构的 Q/K/V 例子。
///
/// 这里不会简单使用全随机数据，而是在随机初始化基础上加入了索引相关的正弦/余弦模式，
/// 这样生成的张量更接近真实模型中“各个 token 有一些特征趋势”的行为，
/// 更利于测试_attention是否在不同 batch/head/seq 维度下保持稳定。
pub fn make_qkv_example(batch: usize, heads: usize, seq: usize, dim: usize, seed: u64) -> (Tensor4, Tensor4, Tensor4) {
    let q = Tensor4::from_random(batch, heads, seq, dim, seed ^ 0xA5A5_A5A5_5A5A_5A5A);
    let k = Tensor4::from_random(batch, heads, seq, dim, seed ^ 0xC3C3_C3C3_3C3C_3C3C);
    let v = Tensor4::from_random(batch, heads, seq, dim, seed ^ 0xD1D1_D1D1_1D1D_1D1D);

    let q = Tensor4::from_fn(batch, heads, seq, dim, |b, h, s, d| {
        let base = q.data[q.index(b, h, s, d)];
        let pattern = ((b + 1) as f32 * (h + 1) as f32 * (s + 1) as f32 * (d + 1) as f32 * 0.17).sin();
        base * 0.8 + pattern * 0.5
    });
    let k = Tensor4::from_fn(batch, heads, seq, dim, |b, h, s, d| {
        let base = k.data[k.index(b, h, s, d)];
        let pattern = ((b + 1) as f32 * (h + 2) as f32 * (s + 3) as f32 * (d + 2) as f32 * 0.19).cos();
        base * 0.85 + pattern * 0.4
    });
    let v = Tensor4::from_fn(batch, heads, seq, dim, |b, h, s, d| {
        let base = v.data[v.index(b, h, s, d)];
        let pattern = ((b + 2) as f32 * (h + 3) as f32 * (s + 4) as f32 * (d + 1) as f32 * 0.12).sin();
        (base + 1.0) * 0.5 + pattern * 0.3
    });

    (q, k, v)
}

/// 用于封装 benchmark 结果的结构体。
///
/// 对于每个输入形状，程序会同时计算：
/// - 朴素实现的耗时；
/// - 分块式 flash attention 的耗时；
/// - 两者之间的数值误差；
/// - 是否满足正确性阈值。
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub batch: usize,
    pub heads: usize,
    pub seq: usize,
    pub dim: usize,
    pub naive_ms: f64,
    pub flash_ms: f64,
    pub max_abs_error: f64,
    pub passed: bool,
}

/// 批量 benchmark 多组输入形状。
///
/// `shapes` 中每个元素表示 `(batch, heads, seq, dim)`，
/// 程序会循环执行 `iterations` 次，统计各个 shape 下的平均耗时和最大误差。
///
/// 这个函数主要用于验证不同维度下的数值表现是否稳定，并为后续优化提供数据基线。
pub fn benchmark_attention_shapes(
    shapes: &[(usize, usize, usize, usize)],
    block_size: usize,
    iterations: usize,
    seed: u64,
) -> Vec<BenchmarkResult> {
    let mut results = Vec::with_capacity(shapes.len());

    for (batch, heads, seq, dim) in shapes {
        let mut max_abs_error = 0.0;
        let mut naive_total = 0.0;
        let mut flash_total = 0.0;

        for i in 0..iterations {
            let (q, k, v) = make_qkv_example(*batch, *heads, *seq, *dim, seed + i as u64);

            let t0 = Instant::now();
            let naive = naive_attention(&q, &k, &v);
            naive_total += t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let fast = flash_attention(&q, &k, &v, block_size);
            flash_total += t1.elapsed().as_secs_f64() * 1000.0;

            for idx in 0..naive.data.len() {
                let err = (naive.data[idx] - fast.data[idx]).abs() as f64;
                if err > max_abs_error {
                    max_abs_error = err;
                }
            }
        }

        results.push(BenchmarkResult {
            batch: *batch,
            heads: *heads,
            seq: *seq,
            dim: *dim,
            naive_ms: naive_total / iterations as f64,
            flash_ms: flash_total / iterations as f64,
            max_abs_error,
            passed: max_abs_error < 1e-4,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::{
        benchmark_attention_shapes, flash_attention, make_qkv_example, naive_attention,
        naive_attention_causal, scaled_dot_product_attention, Tensor4,
    };

    #[test]
    fn basic_small_case_matches_naive() {
        let q = Tensor4::from_vec(
            1,
            1,
            4,
            3,
            vec![
                0.3, 0.2, -0.1,
                0.5, -0.2, 0.1,
                -0.4, 0.7, 0.2,
                0.9, -0.3, 0.4,
            ],
        );
        let k = Tensor4::from_vec(
            1,
            1,
            4,
            3,
            vec![
                0.1, 0.5, 0.2,
                -0.3, 0.2, 0.8,
                0.7, -0.4, 0.1,
                -0.2, 0.6, 0.3,
            ],
        );
        let v = Tensor4::from_vec(
            1,
            1,
            4,
            3,
            vec![
                1.0, 0.0, 0.5,
                -1.0, 0.5, 0.2,
                0.3, 1.0, -0.2,
                0.7, -0.4, 0.9,
            ],
        );

        let actual = scaled_dot_product_attention(&q, &k, &v, None, false, 2);
        let expected = naive_attention(&q, &k, &v);

        for idx in 0..actual.data.len() {
            assert!(
                (actual.data[idx] - expected.data[idx]).abs() < 1e-4,
                "mismatch at index {idx}: actual={} expected={}",
                actual.data[idx],
                expected.data[idx],
            );
        }
    }

    #[test]
    fn multi_batch_multi_head_matches_naive() {
        let (q, k, v) = make_qkv_example(2, 3, 5, 4, 1234);
        let actual = flash_attention(&q, &k, &v, 2);
        let expected = naive_attention(&q, &k, &v);

        for idx in 0..actual.data.len() {
            assert!(
                (actual.data[idx] - expected.data[idx]).abs() < 1e-4,
                "mismatch at index {idx}: actual={} expected={}",
                actual.data[idx],
                expected.data[idx],
            );
        }
    }

    #[test]
    fn causal_attention_matches_reference() {
        let q = Tensor4::from_vec(
            1,
            1,
            3,
            2,
            vec![1.0, 0.0, 0.5, 1.0, -0.5, 0.25],
        );
        let k = Tensor4::from_vec(
            1,
            1,
            3,
            2,
            vec![0.5, 0.2, -0.1, 0.8, 0.2, -0.4],
        );
        let v = Tensor4::from_vec(
            1,
            1,
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let actual = scaled_dot_product_attention(&q, &k, &v, None, true, 2);
        let expected = naive_attention_causal(&q, &k, &v);

        for idx in 0..actual.data.len() {
            assert!(
                (actual.data[idx] - expected.data[idx]).abs() < 1e-4,
                "causal mismatch at index {idx}: actual={} expected={}",
                actual.data[idx],
                expected.data[idx],
            );
        }
    }

    #[test]
    fn uniform_shape_suite_matches_naive() {
        let shapes = [(1, 1, 4, 4), (2, 3, 5, 4), (4, 2, 8, 6)];
        let results = benchmark_attention_shapes(&shapes, 2, 2, 2024);

        assert_eq!(results.len(), shapes.len());
        for r in results {
            assert!(r.passed, "shape {:?} failed: max_abs_error={}", (r.batch, r.heads, r.seq, r.dim), r.max_abs_error);
        }
    }
}
