use std::f32::consts::PI;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Tensor4 {
    pub batch: usize,
    pub heads: usize,
    pub seq: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Tensor4 {
    pub fn new(batch: usize, heads: usize, seq: usize, dim: usize) -> Self {
        Self {
            batch,
            heads,
            seq,
            dim,
            data: vec![0.0; batch * heads * seq * dim],
        }
    }

    pub fn zeros(batch: usize, heads: usize, seq: usize, dim: usize) -> Self {
        Self::new(batch, heads, seq, dim)
    }

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

    pub fn from_random(batch: usize, heads: usize, seq: usize, dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let total = batch * heads * seq * dim;
        let mut data = Vec::with_capacity(total);

        for _ in 0..total {
            let x = rng.next_f32();
            let y = rng.next_f32();
            let z = (-2.0 * x.ln()).sqrt() * (2.0 * PI * y).cos();
            data.push(z * 0.5);
        }

        Self::from_vec(batch, heads, seq, dim, data)
    }

    pub fn index(&self, b: usize, h: usize, s: usize, d: usize) -> usize {
        ((b * self.heads + h) * self.seq + s) * self.dim + d
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() as f64;
        (value / u64::MAX as f64) as f32
    }
}

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
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0;
                let mut o = vec![0.0; dim];

                let max_j = if causal { i + 1 } else { k.seq };

                for block_start in (0..max_j).step_by(block_size) {
                    let block_end = (block_start + block_size).min(max_j);
                    let mut block_m = f32::NEG_INFINITY;
                    let mut block_l = 0.0;
                    let mut block_o = vec![0.0; dim];

                    for j in block_start..block_end {
                        let q_base = ((b * q.heads + h) * q.seq + i) * q.dim;
                        let k_base = ((b * k.heads + h) * k.seq + j) * k.dim;
                        let score = dot_product_qk(&q.data[q_base..q_base + q.dim], &k.data[k_base..k_base + k.dim]) * scale;
                        if score > block_m {
                            block_m = score;
                        }
                    }

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

pub fn flash_attention(q: &Tensor4, k: &Tensor4, v: &Tensor4, block_size: usize) -> Tensor4 {
    scaled_dot_product_attention(q, k, v, None, false, block_size)
}

pub fn flash_attention_cpu(q: &Tensor4, k: &Tensor4, v: &Tensor4, block_size: usize) -> Tensor4 {
    flash_attention(q, k, v, block_size)
}

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
