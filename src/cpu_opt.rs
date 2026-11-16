//! Optimized CPU attention variants used as a fair baseline against the GPU.
//!
//! The original [`crate::attention::scaled_dot_product_attention`] is written
//! for clarity: it recomputes every QK score twice per tile, allocates scratch
//! vectors inside the innermost loop, and runs on a single thread. This module
//! adds three progressively more optimized implementations without changing
//! the numerical definition:
//!
//! 1. [`scaled_dot_product_attention_cached`] — caches each tile's scores,
//!    reuses scratch buffers across rows, hoists row pointers.
//! 2. [`flash_attention_parallel`] — the same kernel split across OS threads
//!    over contiguous query rows (no external dependency).
//! 3. [`naive_attention_parallel`] — the reference algorithm parallelized the
//!    same way, so CPU thread scaling can be measured on both algorithms.
//!
//! Comparing these against the original CPU implementation separates
//! *algorithmic* improvements (caching, allocation reuse) from *hardware*
//! improvements (GPU throughput).

use std::thread;

use crate::attention::Tensor4;

/// Compute the dot product of two equal-length rows.
///
/// Four independent accumulators break the loop-carried add dependency and let
/// the compiler emit a vectorized reduction on aarch64/x86-64.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let chunks = a.len() / 4;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for i in chunks * 4..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[derive(Clone, Copy)]
struct FlashParams {
    dim: usize,
    seq: usize,
    scale: f32,
    causal: bool,
    block_size: usize,
}

fn flash_rows(
    p: &FlashParams,
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
    row_start: usize,
    out_rows: &mut [f32],
) {
    let dim = p.dim;
    let mut scores = vec![0.0f32; p.block_size.min(p.seq)];
    let mut o = vec![0.0f32; dim];

    for (local, out_row) in out_rows.chunks_mut(dim).enumerate() {
        let row = row_start + local;
        let bh = row / p.seq;
        let i = row % p.seq;
        let q_row = &q.data[row * dim..(row + 1) * dim];
        let kv_base = bh * p.seq * dim;
        let max_j = if p.causal { i + 1 } else { p.seq };

        o.fill(0.0);
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;

        let mut start = 0;
        while start < max_j {
            let end = (start + p.block_size).min(max_j);
            let cnt = end - start;

            // Pass 1: QK scores for this tile, cached in `scores`.
            let mut block_m = f32::NEG_INFINITY;
            for jj in 0..cnt {
                let j = start + jj;
                let kb = kv_base + j * dim;
                let s = dot(q_row, &k.data[kb..kb + dim]) * p.scale;
                scores[jj] = s;
                if s > block_m {
                    block_m = s;
                }
            }

            // Merge the tile into the online softmax state. Using `new_m`
            // directly for the exponentials avoids a separate block
            // accumulator and one rescaling multiply per element.
            let new_m = m.max(block_m);
            let alpha = if m == f32::NEG_INFINITY {
                0.0
            } else {
                (m - new_m).exp()
            };
            if alpha == 0.0 {
                o.fill(0.0);
                l = 0.0;
            } else if alpha != 1.0 {
                for x in o.iter_mut() {
                    *x *= alpha;
                }
                l *= alpha;
            }

            // Pass 2: P·V with the stabilized weights.
            for jj in 0..cnt {
                let j = start + jj;
                let pj = (scores[jj] - new_m).exp();
                l += pj;
                let vb = kv_base + j * dim;
                let v_row = &v.data[vb..vb + dim];
                for d in 0..dim {
                    o[d] += pj * v_row[d];
                }
            }

            m = new_m;
            start = end;
        }

        let inv_l = 1.0 / l;
        for d in 0..dim {
            out_row[d] = o[d] * inv_l;
        }
    }
}

fn naive_rows(
    dim: usize,
    seq: usize,
    scale: f32,
    causal: bool,
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
    row_start: usize,
    out_rows: &mut [f32],
) {
    let mut scores = vec![0.0f32; seq];

    for (local, out_row) in out_rows.chunks_mut(dim).enumerate() {
        let row = row_start + local;
        let bh = row / seq;
        let i = row % seq;
        let q_row = &q.data[row * dim..(row + 1) * dim];
        let kv_base = bh * seq * dim;
        let max_j = if causal { i + 1 } else { seq };

        let mut mx = f32::NEG_INFINITY;
        for j in 0..max_j {
            let kb = kv_base + j * dim;
            let s = dot(q_row, &k.data[kb..kb + dim]) * scale;
            scores[j] = s;
            if s > mx {
                mx = s;
            }
        }

        let mut sum = 0.0f32;
        for score in scores.iter_mut().take(max_j) {
            let e = (*score - mx).exp();
            *score = e;
            sum += e;
        }

        let inv_sum = 1.0 / sum;
        for d in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..max_j {
                acc += scores[j] * v.data[kv_base + j * dim + d];
            }
            out_row[d] = acc * inv_sum;
        }
    }
}

fn resolve_threads(requested: usize, rows: usize) -> usize {
    let n = if requested == 0 {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        requested
    };
    n.clamp(1, rows.max(1))
}

fn parallel_rows<F>(out: &mut Tensor4, rows: usize, num_threads: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let dim = out.dim;
    let threads = resolve_threads(num_threads, rows);
    let chunk_rows = rows.div_ceil(threads);
    thread::scope(|scope| {
        for (chunk_idx, out_chunk) in out.data.chunks_mut(chunk_rows * dim).enumerate() {
            let start_row = chunk_idx * chunk_rows;
            let f = &f;
            scope.spawn(move || f(start_row, out_chunk));
        }
    });
}

/// Single-thread Flash Attention without redundant QK work.
pub fn scaled_dot_product_attention_cached(
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
    assert_eq!(q.seq, k.seq);
    assert_eq!(q.seq, v.seq);
    assert_eq!(q.dim, k.dim);
    assert_eq!(q.dim, v.dim);
    assert!(block_size > 0);

    let dim = q.dim;
    let scale = scale.unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
    let mut out = Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    let p = FlashParams {
        dim,
        seq: q.seq,
        scale,
        causal,
        block_size,
    };
    let rows = q.batch * q.heads * q.seq;
    flash_rows(&p, q, k, v, 0, &mut out.data);
    debug_assert_eq!(out.data.len(), rows * dim);
    out
}

/// Multi-threaded Flash Attention with cached QK tiles.
pub fn flash_attention_parallel(
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
    block_size: usize,
    num_threads: usize,
) -> Tensor4 {
    scaled_dot_product_attention_parallel(q, k, v, None, false, block_size, num_threads)
}

/// Multi-threaded Flash Attention with explicit scale/causal controls.
pub fn scaled_dot_product_attention_parallel(
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
    scale: Option<f32>,
    causal: bool,
    block_size: usize,
    num_threads: usize,
) -> Tensor4 {
    assert_eq!(q.batch, k.batch);
    assert_eq!(q.heads, k.heads);
    assert_eq!(q.batch, v.batch);
    assert_eq!(q.heads, v.heads);
    assert_eq!(q.seq, k.seq);
    assert_eq!(q.seq, v.seq);
    assert_eq!(q.dim, k.dim);
    assert_eq!(q.dim, v.dim);
    assert!(block_size > 0);

    let dim = q.dim;
    let rows = q.batch * q.heads * q.seq;
    if rows == 0 {
        return Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    }
    let scale = scale.unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
    let mut out = Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    let p = FlashParams {
        dim,
        seq: q.seq,
        scale,
        causal,
        block_size,
    };
    parallel_rows(&mut out, rows, num_threads, |start, chunk| {
        flash_rows(&p, q, k, v, start, chunk);
    });
    out
}

/// Multi-threaded reference (materialized-per-row) attention.
pub fn naive_attention_parallel(
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
    causal: bool,
    num_threads: usize,
) -> Tensor4 {
    assert_eq!(q.batch, k.batch);
    assert_eq!(q.heads, k.heads);
    assert_eq!(q.batch, v.batch);
    assert_eq!(q.heads, v.heads);
    assert_eq!(q.seq, k.seq);
    assert_eq!(q.seq, v.seq);
    assert_eq!(q.dim, k.dim);
    assert_eq!(q.dim, v.dim);

    let rows = q.batch * q.heads * q.seq;
    if rows == 0 {
        return Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    }
    let dim = q.dim;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = Tensor4::new(q.batch, q.heads, q.seq, q.dim);
    parallel_rows(&mut out, rows, num_threads, |start, chunk| {
        naive_rows(dim, q.seq, scale, causal, q, k, v, start, chunk);
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{
        flash_attention, make_qkv_example, naive_attention, naive_attention_causal,
    };

    fn max_abs_diff(a: &Tensor4, b: &Tensor4) -> f32 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn cached_matches_reference_for_all_modes() {
        for &(b, h, s, d) in &[(1, 1, 4, 4), (2, 3, 17, 8), (1, 2, 33, 16), (2, 2, 64, 32)] {
            let (q, k, v) = make_qkv_example(b, h, s, d, 42);
            for &causal in &[false, true] {
                for &block in &[1usize, 2, 7, 16, 64] {
                    let got = scaled_dot_product_attention_cached(&q, &k, &v, None, causal, block);
                    let want = if causal {
                        naive_attention_causal(&q, &k, &v)
                    } else {
                        naive_attention(&q, &k, &v)
                    };
                    let diff = max_abs_diff(&got, &want);
                    assert!(
                        diff < 1e-4,
                        "cached mismatch shape={:?} causal={causal} block={block} diff={diff}",
                        (b, h, s, d)
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_matches_single_thread() {
        let (q, k, v) = make_qkv_example(2, 4, 41, 16, 123);
        for &threads in &[1usize, 2, 3, 8, 16] {
            for &causal in &[false, true] {
                let got =
                    scaled_dot_product_attention_parallel(&q, &k, &v, None, causal, 8, threads);
                let want = scaled_dot_product_attention_cached(&q, &k, &v, None, causal, 8);
                assert!(max_abs_diff(&got, &want) < 1e-5);
            }
        }
    }

    #[test]
    fn parallel_naive_matches_reference() {
        let (q, k, v) = make_qkv_example(2, 3, 23, 16, 5);
        let got = naive_attention_parallel(&q, &k, &v, false, 4);
        let want = naive_attention(&q, &k, &v);
        assert!(max_abs_diff(&got, &want) < 1e-5);
        let got = naive_attention_parallel(&q, &k, &v, true, 4);
        let want = naive_attention_causal(&q, &k, &v);
        assert!(max_abs_diff(&got, &want) < 1e-5);
    }

    #[test]
    fn cached_flash_is_no_worse_than_baseline_on_small_case() {
        // Not a timing assertion, just a smoke test that the optimized entry
        // points agree with the original implementation.
        let (q, k, v) = make_qkv_example(2, 2, 19, 16, 99);
        let a = flash_attention(&q, &k, &v, 4);
        let b = scaled_dot_product_attention_cached(&q, &k, &v, None, false, 4);
        assert!(max_abs_diff(&a, &b) < 1e-4);
    }
}
