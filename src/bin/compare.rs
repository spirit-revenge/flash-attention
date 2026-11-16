//! CPU vs GPU comparison harness for Flash Attention.
//!
//! Run with:
//!   cargo run --release --features metal --bin compare
//!   cargo run --release --features metal --bin compare -- --quick
//!
//! The harness writes raw measurements to `results/*.csv`, device/host metadata
//! to `results/meta.txt`, and prints a compact summary. `scripts/report.py`
//! turns the CSVs into charts and markdown tables.
//!
//! Measurement protocol
//! --------------------
//! * CPU: one warm-up call, then `iterations` timed calls; median reported.
//! * GPU: `GpuAttention::benchmark_*` reports
//!   - per-call latency (dispatch + wait every iteration),
//!   - amortized kernel time (N dispatches in one command buffer),
//!   - end-to-end time (upload + dispatch + download).
//!   Median per-call is used for the latency column, amortized time for
//!   throughput. The machine is a fanless MacBook Air, so the harness keeps GPU
//!   bursts short and reports min/max alongside the median.
//!
//! All error metrics are computed against the original single-threaded CPU
//! `naive_attention` (the same reference used by the unit tests).

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use flash_attention::attention::{
    Tensor4, make_qkv_example, naive_attention, naive_attention_causal,
    scaled_dot_product_attention,
};
use flash_attention::cpu_opt::{
    naive_attention_parallel, scaled_dot_product_attention_cached,
    scaled_dot_product_attention_parallel,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use flash_attention::gpu::{AttnOptions, GpuAttention};

const CPU_BLOCK: usize = 64;

#[derive(Clone, Copy, Debug)]
struct Case {
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    causal: bool,
}

impl Case {
    fn bh(&self) -> usize {
        self.batch * self.heads
    }

    fn flops(&self) -> f64 {
        let (b, h, s, d) = (
            self.batch as f64,
            self.heads as f64,
            self.seq as f64,
            self.dim as f64,
        );
        if self.causal {
            // sum_j (i+1) = S(S+1)/2 keys, times 2 matmuls * 2 flops
            2.0 * b * h * s * (s + 1.0) * d
        } else {
            4.0 * b * h * s * s * d
        }
    }

    fn label(&self) -> String {
        format!("B{}H{}S{}D{}", self.batch, self.heads, self.seq, self.dim)
    }
}

#[derive(Clone, Debug)]
struct Row {
    suite: String,
    implementation: String,
    backend: String,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    causal: bool,
    latency_ms: f64,
    throughput_ms: f64,
    e2e_ms: Option<f64>,
    tflops: f64,
    max_abs_err: f64,
    mean_abs_err: f64,
    rel_l2: f64,
    iterations: usize,
    gpu_bytes: u64,
    threads: usize,
    block_size: usize,
    query_block: usize,
    note: String,
}

impl Row {
    fn new(
        suite: &str,
        implementation: &str,
        backend: &str,
        case: Case,
        latency_ms: f64,
        throughput_ms: f64,
        e2e_ms: Option<f64>,
        iterations: usize,
        block_size: usize,
        query_block: usize,
        threads: usize,
    ) -> Self {
        Self {
            suite: suite.to_string(),
            implementation: implementation.to_string(),
            backend: backend.to_string(),
            batch: case.batch,
            heads: case.heads,
            seq: case.seq,
            dim: case.dim,
            causal: case.causal,
            latency_ms,
            throughput_ms,
            e2e_ms,
            tflops: case.flops() / throughput_ms / 1e9,
            max_abs_err: 0.0,
            mean_abs_err: 0.0,
            rel_l2: 0.0,
            iterations,
            gpu_bytes: 0,
            threads,
            block_size,
            query_block,
            note: String::new(),
        }
    }
}

struct Harness {
    rows: Vec<Row>,
    gpu: Option<GpuAttention>,
    gpu_note: String,
    available_threads: usize,
    quick: bool,
    naive_gpu_budget_bytes: u64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let out_dir = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "results".to_string());
    let out_dir = PathBuf::from(out_dir);
    fs::create_dir_all(&out_dir).expect("create results dir");

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let (gpu, gpu_note) = match GpuAttention::new() {
        Ok(g) => (Some(g), String::new()),
        Err(e) => (None, format!("GPU unavailable: {e}")),
    };

    let mut h = Harness {
        rows: Vec::new(),
        gpu,
        gpu_note,
        available_threads,
        quick,
        naive_gpu_budget_bytes: if quick { 256 << 20 } else { 2 << 30 },
    };

    h.print_header();
    h.run_scaling_d64();
    h.run_scaling_d128();
    h.run_causal();
    h.run_batch_heads();
    h.run_block_sweep();
    h.run_thread_scaling();
    h.run_edge_cases();
    h.run_transfer();
    h.write_startup(&out_dir);

    h.write_csvs(&out_dir);
    h.write_meta(&out_dir);
    h.print_summary();

    println!("\nRaw results written to {}", out_dir.display());
}

impl Harness {
    fn print_header(&self) {
        println!("=== Flash Attention CPU vs GPU comparison ===");
        println!(
            "host threads: {}, mode: {}",
            self.available_threads,
            if self.quick { "quick" } else { "full" }
        );
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &self.gpu {
            let info = gpu.info();
            println!(
                "GPU: {} | unified={} | tgmem={} B | max buffer={:.2} GiB",
                info.name,
                info.unified_memory,
                info.max_threadgroup_memory,
                info.max_buffer_length as f64 / (1u64 << 30) as f64
            );
        }
        if !self.gpu_note.is_empty() {
            println!("{}", self.gpu_note);
        }
    }

    fn scaling_cases_d64(&self) -> Vec<Case> {
        let seqs: &[usize] = if self.quick {
            &[128, 256, 512, 1024]
        } else {
            &[128, 256, 512, 1024, 2048, 4096]
        };
        seqs.iter()
            .map(|&s| Case {
                batch: 1,
                heads: 8,
                seq: s,
                dim: 64,
                causal: false,
            })
            .collect()
    }

    fn run_scaling_d64(&mut self) {
        println!("\n[1/8] sequence-length scaling, dim=64, B*H=8");
        let cases = self.scaling_cases_d64();
        self.run_case_suite("scaling_d64", &cases);
    }

    fn run_scaling_d128(&mut self) {
        println!("\n[2/8] sequence-length scaling, dim=128, B*H=8");
        let seqs: &[usize] = if self.quick {
            &[256, 512]
        } else {
            &[256, 512, 1024, 2048]
        };
        let cases: Vec<Case> = seqs
            .iter()
            .map(|&s| Case {
                batch: 1,
                heads: 8,
                seq: s,
                dim: 128,
                causal: false,
            })
            .collect();
        self.run_case_suite("scaling_d128", &cases);
    }

    fn run_causal(&mut self) {
        println!("\n[3/8] causal masking, dim=64, B*H=8");
        let seqs: &[usize] = if self.quick {
            &[512, 1024]
        } else {
            &[1024, 2048, 4096]
        };
        let cases: Vec<Case> = seqs
            .iter()
            .map(|&s| Case {
                batch: 1,
                heads: 8,
                seq: s,
                dim: 64,
                causal: true,
            })
            .collect();
        self.run_case_suite("causal", &cases);
    }

    fn run_batch_heads(&mut self) {
        println!("\n[4/8] batch/head scaling, S=1024, D=64");
        let shapes = if self.quick {
            vec![(1, 1), (1, 4), (1, 8)]
        } else {
            vec![(1, 1), (1, 4), (1, 8), (2, 8), (4, 4)]
        };
        let cases: Vec<Case> = shapes
            .into_iter()
            .map(|(b, h)| Case {
                batch: b,
                heads: h,
                seq: 1024,
                dim: 64,
                causal: false,
            })
            .collect();
        self.run_case_suite("batch_heads", &cases);
    }

    /// Run every available implementation on each case, with a shared CPU
    /// reference. GPU rows are measured before CPU rows to keep GPU bursts
    /// away from long CPU-heavy runs (thermal fairness on a fanless laptop).
    fn run_case_suite(&mut self, suite: &str, cases: &[Case]) {
        for (idx, &case) in cases.iter().enumerate() {
            let seed = 2024 + idx as u64;
            let (q, k, v) = make_qkv_example(case.batch, case.heads, case.seq, case.dim, seed);

            // Reference: the original single-threaded naive implementation.
            let ref_t0 = Instant::now();
            let reference = if case.causal {
                naive_attention_causal(&q, &k, &v)
            } else {
                naive_attention(&q, &k, &v)
            };
            let ref_ms = ref_t0.elapsed().as_secs_f64() * 1e3;
            let ref_perf = case.flops() / ref_ms / 1e9;

            // The reference itself is a measured implementation, so record it.
            let mut ref_row = Row::new(
                suite,
                "cpu_naive",
                "cpu",
                case,
                ref_ms,
                ref_ms,
                None,
                1,
                0,
                0,
                1,
            );
            ref_row.tflops = ref_perf;
            self.rows.push(ref_row);

            // ---- GPU implementations -------------------------------------
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Some(gpu) = &self.gpu {
                let opts = AttnOptions {
                    causal: case.causal,
                    block_size: 32,
                    query_block: 32,
                    ..Default::default()
                };
                let iters = if case.seq >= 4096 { 12 } else { 20 };

                // Correctness pass (also warms up the compiled pipeline).
                match gpu.flash_attention(&q, &k, &v, opts) {
                    Ok(got) => {
                        let bench = gpu.benchmark_flash(&q, &k, &v, opts, iters).unwrap();
                        let mut row = Row::new(
                            suite,
                            "gpu_flash",
                            "gpu",
                            case,
                            bench.per_call_ms,
                            bench.amortized_ms,
                            Some(bench.end_to_end_ms),
                            iters,
                            opts.block_size,
                            opts.query_block,
                            0,
                        );
                        row.gpu_bytes = bench.gpu_bytes;
                        let (mx, mean, rel) = error_stats(&got, &reference);
                        row.max_abs_err = mx;
                        row.mean_abs_err = mean;
                        row.rel_l2 = rel;
                        row.note = format!(
                            "groups={:?} threads/group={} tgmem={}",
                            bench.groups, bench.threads_per_group, bench.threadgroup_memory_bytes
                        );
                        self.rows.push(row);
                    }
                    Err(e) => println!("  {} gpu_flash failed: {e}", case.label()),
                }

                // Naive GPU baseline, only when the O(S^2) score matrix is
                // within the configured memory budget.
                let score_bytes = (case.bh() * case.seq * case.seq * 4) as u64;
                if score_bytes <= self.naive_gpu_budget_bytes {
                    let niters = if case.seq >= 2048 { 4 } else { 10 };
                    match gpu.naive_attention(&q, &k, &v, opts) {
                        Ok(got) => {
                            let bench = gpu.benchmark_naive(&q, &k, &v, opts, niters).unwrap();
                            let mut row = Row::new(
                                suite,
                                "gpu_naive",
                                "gpu",
                                case,
                                bench.per_call_ms,
                                bench.amortized_ms,
                                Some(bench.end_to_end_ms),
                                niters,
                                0,
                                0,
                                0,
                            );
                            row.gpu_bytes = bench.gpu_bytes;
                            let (mx, mean, rel) = error_stats(&got, &reference);
                            row.max_abs_err = mx;
                            row.mean_abs_err = mean;
                            row.rel_l2 = rel;
                            row.note = format!(
                                "materialized scores {:.1} MiB",
                                score_bytes as f64 / (1u64 << 20) as f64
                            );
                            self.rows.push(row);
                        }
                        Err(e) => println!("  {} gpu_naive failed: {e}", case.label()),
                    }
                } else {
                    self.rows.push({
                        let mut row = Row::new(
                            suite,
                            "gpu_naive",
                            "gpu",
                            case,
                            f64::NAN,
                            f64::NAN,
                            None,
                            0,
                            0,
                            0,
                            0,
                        );
                        row.note = format!(
                            "skipped: score matrix {:.2} GiB exceeds budget {:.2} GiB",
                            score_bytes as f64 / (1u64 << 30) as f64,
                            self.naive_gpu_budget_bytes as f64 / (1u64 << 30) as f64
                        );
                        row
                    });
                }
            }

            // ---- CPU implementations -------------------------------------
            let target_seconds = 0.25;

            let baseline_iters = cpu_iterations(case.flops(), 1.0e10, target_seconds);
            let (ms, _, out) = time_cpu(
                || scaled_dot_product_attention(&q, &k, &v, None, case.causal, CPU_BLOCK),
                baseline_iters,
            );
            self.push_cpu_row(
                suite,
                "cpu_flash_1t",
                case,
                ms,
                baseline_iters,
                CPU_BLOCK,
                1,
                &out,
                &reference,
            );

            let cached_iters = cpu_iterations(case.flops(), 1.4e10, target_seconds);
            let (ms, _, out) = time_cpu(
                || scaled_dot_product_attention_cached(&q, &k, &v, None, case.causal, CPU_BLOCK),
                cached_iters,
            );
            self.push_cpu_row(
                suite,
                "cpu_cached_1t",
                case,
                ms,
                cached_iters,
                CPU_BLOCK,
                1,
                &out,
                &reference,
            );

            let par_iters = cpu_iterations(case.flops(), 6.0e10, target_seconds);
            let (ms, _, out) = time_cpu(
                || {
                    scaled_dot_product_attention_parallel(
                        &q,
                        &k,
                        &v,
                        None,
                        case.causal,
                        CPU_BLOCK,
                        0,
                    )
                },
                par_iters,
            );
            self.push_cpu_row(
                suite,
                "cpu_parallel",
                case,
                ms,
                par_iters,
                CPU_BLOCK,
                self.available_threads,
                &out,
                &reference,
            );

            println!("  {:<24} reference {:.3} ms", case.label(), ref_ms);
        }
    }

    fn push_cpu_row(
        &mut self,
        suite: &str,
        name: &str,
        case: Case,
        ms: f64,
        iterations: usize,
        block_size: usize,
        threads: usize,
        got: &Tensor4,
        reference: &Tensor4,
    ) {
        let mut row = Row::new(
            suite, name, "cpu", case, ms, ms, None, iterations, block_size, block_size, threads,
        );
        let (mx, mean, rel) = error_stats(got, reference);
        row.max_abs_err = mx;
        row.mean_abs_err = mean;
        row.rel_l2 = rel;
        self.rows.push(row);
    }

    fn run_block_sweep(&mut self) {
        println!("\n[5/8] block-size sweep, S=1024, D=64, B*H=8");
        let case = Case {
            batch: 1,
            heads: 8,
            seq: 1024,
            dim: 64,
            causal: false,
        };
        let (q, k, v) = make_qkv_example(case.batch, case.heads, case.seq, case.dim, 7);
        let reference = naive_attention(&q, &k, &v);
        let flops = case.flops();
        let cpu_blocks: &[usize] = if self.quick {
            &[32, 64, 256]
        } else {
            &[16, 32, 64, 128, 256, 512, 1024]
        };

        for &block in cpu_blocks {
            let iters = cpu_iterations(flops, 1.4e10, 0.15);
            let (ms, _, out) = time_cpu(
                || scaled_dot_product_attention_cached(&q, &k, &v, None, false, block),
                iters,
            );
            let mut row = Row::new(
                "block_sweep",
                &format!("cpu_cached_b{block}"),
                "cpu",
                case,
                ms,
                ms,
                None,
                iters,
                block,
                block,
                1,
            );
            let (mx, mean, rel) = error_stats(&out, &reference);
            row.max_abs_err = mx;
            row.mean_abs_err = mean;
            row.rel_l2 = rel;
            self.rows.push(row);

            let iters = cpu_iterations(flops, 6.0e10, 0.15);
            let (ms, _, out) = time_cpu(
                || scaled_dot_product_attention_parallel(&q, &k, &v, None, false, block, 0),
                iters,
            );
            let mut row = Row::new(
                "block_sweep",
                &format!("cpu_parallel_b{block}"),
                "cpu",
                case,
                ms,
                ms,
                None,
                iters,
                block,
                block,
                self.available_threads,
            );
            let (mx, mean, rel) = error_stats(&out, &reference);
            row.max_abs_err = mx;
            row.mean_abs_err = mean;
            row.rel_l2 = rel;
            self.rows.push(row);
        }

        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &self.gpu {
            for &dim in &[64usize, 128usize] {
                let case = Case {
                    batch: 1,
                    heads: 8,
                    seq: 1024,
                    dim,
                    causal: false,
                };
                let (q, k, v) = make_qkv_example(case.batch, case.heads, case.seq, case.dim, 7);
                let reference = naive_attention(&q, &k, &v);
                let max_bc = gpu.max_block_size(dim);
                let bcs: Vec<usize> = if self.quick {
                    vec![16, 32]
                } else {
                    vec![16, 32, 48, 64]
                };
                let brs: &[usize] = if self.quick { &[32] } else { &[16, 32, 64] };
                for &bc in &bcs {
                    if bc > max_bc {
                        continue;
                    }
                    for &br in brs {
                        let opts = AttnOptions {
                            block_size: bc,
                            query_block: br,
                            ..Default::default()
                        };
                        let lanes = if dim <= 64 { 4 } else { 8 };
                        if br * lanes > 1024 || (br * lanes) % 32 != 0 {
                            continue;
                        }
                        match gpu.flash_attention(&q, &k, &v, opts) {
                            Ok(got) => {
                                let bench = gpu.benchmark_flash(&q, &k, &v, opts, 12).unwrap();
                                let mut row = Row::new(
                                    "block_sweep",
                                    &format!("gpu_flash_bc{bc}_br{br}"),
                                    "gpu",
                                    case,
                                    bench.per_call_ms,
                                    bench.amortized_ms,
                                    Some(bench.end_to_end_ms),
                                    12,
                                    bc,
                                    br,
                                    0,
                                );
                                row.gpu_bytes = bench.gpu_bytes;
                                let (mx, mean, rel) = error_stats(&got, &reference);
                                row.max_abs_err = mx;
                                row.mean_abs_err = mean;
                                row.rel_l2 = rel;
                                row.note = format!(
                                    "tgmem={} threads/group={}",
                                    bench.threadgroup_memory_bytes, bench.threads_per_group
                                );
                                self.rows.push(row);
                            }
                            Err(e) => println!("  gpu bc={bc} br={br} failed: {e}"),
                        }
                    }
                }
            }
        }
    }

    fn run_thread_scaling(&mut self) {
        println!("\n[6/8] CPU thread scaling, S=1024/2048, D=64, B*H=8");
        let seqs: &[usize] = if self.quick { &[1024] } else { &[1024, 2048] };
        let thread_counts: &[usize] = if self.quick {
            &[1, 4, 8]
        } else {
            &[1, 2, 4, 6, 8, 10]
        };
        for &seq in seqs {
            let case = Case {
                batch: 1,
                heads: 8,
                seq,
                dim: 64,
                causal: false,
            };
            let (q, k, v) = make_qkv_example(1, 8, seq, 64, 31 + seq as u64);
            let reference = naive_attention(&q, &k, &v);
            for &threads in thread_counts {
                if threads > self.available_threads {
                    continue;
                }
                for (name, naive) in [("cpu_flash", false), ("cpu_naive", true)] {
                    let rate = if naive { 5e9 } else { 6e10 };
                    let iters = cpu_iterations(case.flops(), rate, 0.15);
                    let (ms, _, out) = time_cpu(
                        || {
                            if naive {
                                naive_attention_parallel(&q, &k, &v, false, threads)
                            } else {
                                scaled_dot_product_attention_parallel(
                                    &q, &k, &v, None, false, CPU_BLOCK, threads,
                                )
                            }
                        },
                        iters,
                    );
                    let impl_name = format!("{name}_p{threads}");
                    let mut row = Row::new(
                        "thread_scaling",
                        &impl_name,
                        "cpu",
                        case,
                        ms,
                        ms,
                        None,
                        iters,
                        CPU_BLOCK,
                        CPU_BLOCK,
                        threads,
                    );
                    let (mx, mean, rel) = error_stats(&out, &reference);
                    row.max_abs_err = mx;
                    row.mean_abs_err = mean;
                    row.rel_l2 = rel;
                    self.rows.push(row);
                }
            }
        }
    }

    fn run_edge_cases(&mut self) {
        println!("\n[7/8] odd shapes (correctness + tail handling)");
        let cases = [
            Case {
                batch: 1,
                heads: 3,
                seq: 777,
                dim: 64,
                causal: false,
            },
            Case {
                batch: 3,
                heads: 5,
                seq: 333,
                dim: 32,
                causal: true,
            },
        ];
        for (idx, &case) in cases.iter().enumerate() {
            let (q, k, v) =
                make_qkv_example(case.batch, case.heads, case.seq, case.dim, 900 + idx as u64);
            let naive_t0 = Instant::now();
            let reference = if case.causal {
                naive_attention_causal(&q, &k, &v)
            } else {
                naive_attention(&q, &k, &v)
            };
            let naive_ms = naive_t0.elapsed().as_secs_f64() * 1e3;
            let mut row = Row::new(
                "edge",
                "cpu_naive",
                "cpu",
                case,
                naive_ms,
                naive_ms,
                None,
                1,
                0,
                0,
                1,
            );
            row.tflops = case.flops() / naive_ms / 1e9;
            self.rows.push(row);

            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Some(gpu) = &self.gpu {
                let opts = AttnOptions {
                    causal: case.causal,
                    block_size: 32,
                    query_block: 32,
                    ..Default::default()
                };
                match gpu.flash_attention(&q, &k, &v, opts) {
                    Ok(got) => {
                        let bench = gpu.benchmark_flash(&q, &k, &v, opts, 12).unwrap();
                        let mut row = Row::new(
                            "edge",
                            "gpu_flash",
                            "gpu",
                            case,
                            bench.per_call_ms,
                            bench.amortized_ms,
                            Some(bench.end_to_end_ms),
                            12,
                            opts.block_size,
                            opts.query_block,
                            0,
                        );
                        row.gpu_bytes = bench.gpu_bytes;
                        let (mx, mean, rel) = error_stats(&got, &reference);
                        row.max_abs_err = mx;
                        row.mean_abs_err = mean;
                        row.rel_l2 = rel;
                        self.rows.push(row);
                    }
                    Err(e) => println!("  {} gpu_flash failed: {e}", case.label()),
                }
            }

            let (ms, _, out) = time_cpu(
                || scaled_dot_product_attention_cached(&q, &k, &v, None, case.causal, CPU_BLOCK),
                2,
            );
            self.push_cpu_row(
                "edge",
                "cpu_cached_1t",
                case,
                ms,
                2,
                CPU_BLOCK,
                1,
                &out,
                &reference,
            );

            let (ms, _, out) = time_cpu(
                || {
                    scaled_dot_product_attention_parallel(
                        &q,
                        &k,
                        &v,
                        None,
                        case.causal,
                        CPU_BLOCK,
                        0,
                    )
                },
                2,
            );
            self.push_cpu_row(
                "edge",
                "cpu_parallel",
                case,
                ms,
                2,
                CPU_BLOCK,
                self.available_threads,
                &out,
                &reference,
            );
        }
    }

    fn run_transfer(&mut self) {
        println!("\n[8/8] host/device transfer bandwidth");
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &self.gpu {
            let sizes: &[u64] = if self.quick {
                &[1 << 20, 16 << 20]
            } else {
                &[1 << 20, 8 << 20, 64 << 20, 256 << 20]
            };
            for &size in sizes {
                let iters = if size >= 256 << 20 { 8 } else { 20 };
                match gpu.transfer_benchmark(size, iters) {
                    Ok(t) => {
                        println!(
                            "  {:>6} MiB | upload {:6.1} GB/s | download {:6.1} GB/s | device {:6.1} GB/s",
                            size >> 20,
                            t.upload_gbps,
                            t.download_gbps,
                            t.device_copy_gbps
                        );
                        self.rows.push(Row {
                            suite: "transfer".into(),
                            implementation: format!("transfer_{}mib", size >> 20),
                            backend: "gpu".into(),
                            batch: 0,
                            heads: 0,
                            seq: 0,
                            dim: 0,
                            causal: false,
                            latency_ms: f64::NAN,
                            throughput_ms: f64::NAN,
                            e2e_ms: None,
                            tflops: t.upload_gbps,
                            max_abs_err: t.download_gbps,
                            mean_abs_err: t.device_copy_gbps,
                            rel_l2: f64::NAN,
                            iterations: iters,
                            gpu_bytes: size,
                            threads: 0,
                            block_size: 0,
                            query_block: 0,
                            note: "tflops=upload GB/s, max_abs_err=download GB/s, mean_abs_err=device copy GB/s".into(),
                        });
                    }
                    Err(e) => println!("  transfer {} failed: {e}", size),
                }
            }
        }
    }

    fn write_startup(&mut self, out_dir: &Path) {
        let mut file = match File::create(out_dir.join("startup.csv")) {
            Ok(f) => BufWriter::new(f),
            Err(e) => {
                eprintln!("cannot write startup.csv: {e}");
                return;
            }
        };
        let _ = writeln!(file, "item,value_ms,note");
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &self.gpu {
            let _ = writeln!(
                file,
                "device_and_queue,{:.4},GpuAttention::new (device + command queue)",
                gpu.info().startup_ms
            );
            for &dim in &[64usize, 128usize] {
                match gpu.warmup_dim(dim) {
                    Ok(ms) => {
                        let _ = writeln!(
                            file,
                            "shader_compile_d{dim},{ms:.4},MSL compile + 4 compute pipelines"
                        );
                    }
                    Err(e) => println!("warmup_dim({dim}) failed: {e}"),
                }
            }
            // Cold first call including compilation, on a fresh instance.
            if let Ok(fresh) = GpuAttention::new() {
                let (q, k, v) = make_qkv_example(1, 8, 256, 64, 1);
                let opts = AttnOptions::default();
                let t0 = Instant::now();
                let _ = std::hint::black_box(fresh.flash_attention(&q, &k, &v, opts));
                let cold = t0.elapsed().as_secs_f64() * 1e3;
                let t0 = Instant::now();
                let _ = std::hint::black_box(fresh.flash_attention(&q, &k, &v, opts));
                let warm = t0.elapsed().as_secs_f64() * 1e3;
                let _ = writeln!(
                    file,
                    "first_call_d64,{cold:.4},includes MSL compile + pipeline creation"
                );
                let _ = writeln!(file, "second_call_d64,{warm:.4},warm end-to-end");
            }
        }
    }

    fn write_csvs(&self, out_dir: &Path) {
        let mut main_rows: Vec<&Row> = self
            .rows
            .iter()
            .filter(|r| {
                matches!(
                    r.suite.as_str(),
                    "scaling_d64" | "scaling_d128" | "causal" | "batch_heads" | "edge"
                )
            })
            .collect();
        main_rows.sort_by(|a, b| {
            (&a.suite, a.seq, a.dim, &a.implementation, a.causal).cmp(&(
                &b.suite,
                b.seq,
                b.dim,
                &b.implementation,
                b.causal,
            ))
        });
        write_rows(&out_dir.join("timings.csv"), &main_rows);

        let block_rows: Vec<&Row> = self
            .rows
            .iter()
            .filter(|r| r.suite == "block_sweep")
            .collect();
        write_rows(&out_dir.join("blocksize.csv"), &block_rows);

        let thread_rows: Vec<&Row> = self
            .rows
            .iter()
            .filter(|r| r.suite == "thread_scaling")
            .collect();
        write_rows(&out_dir.join("threads.csv"), &thread_rows);

        let transfer_rows: Vec<&Row> = self.rows.iter().filter(|r| r.suite == "transfer").collect();
        write_transfer_csv(&out_dir.join("transfer.csv"), &transfer_rows);
    }

    fn write_meta(&self, out_dir: &Path) {
        let path = out_dir.join("meta.txt");
        let mut file = match File::create(&path) {
            Ok(f) => BufWriter::new(f),
            Err(e) => {
                eprintln!("cannot write meta: {e}");
                return;
            }
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "timestamp_unix={now}");
        let _ = writeln!(file, "mode={}", if self.quick { "quick" } else { "full" });
        let _ = writeln!(file, "available_threads={}", self.available_threads);
        let rustc = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_else(|| "unknown".into());
        let _ = writeln!(file, "rustc={}", rustc.trim());
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &self.gpu {
            let info = gpu.info();
            let _ = writeln!(file, "gpu_name={}", info.name);
            let _ = writeln!(file, "gpu_unified_memory={}", info.unified_memory);
            let _ = writeln!(
                file,
                "gpu_max_threadgroup_memory={}",
                info.max_threadgroup_memory
            );
            let _ = writeln!(
                file,
                "gpu_max_threads_per_threadgroup={}",
                info.max_threads_per_threadgroup
            );
            let _ = writeln!(file, "gpu_max_buffer_length={}", info.max_buffer_length);
            let _ = writeln!(
                file,
                "gpu_recommended_max_working_set={}",
                info.recommended_max_working_set
            );
            let _ = writeln!(file, "gpu_startup_ms={:.4}", info.startup_ms);
            let compiled = gpu.compiled_dims();
            let _ = writeln!(file, "gpu_compiled_dims={compiled:?}");
        }
        if !self.gpu_note.is_empty() {
            let _ = writeln!(file, "gpu_note={}", self.gpu_note);
        }
        let _ = writeln!(
            file,
            "note=Apple M4 in a fanless MacBook Air; GPU numbers are medians of short bursts, expect thermal variance"
        );
    }

    fn print_summary(&self) {
        println!("\n=== summary (sequence-length scaling, dim=64, non-causal) ===");
        println!(
            "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
            "seq", "cpu_flash_1t", "cpu_parallel", "gpu_flash", "gpu_naive"
        );
        let mut seqs: Vec<usize> = self
            .rows
            .iter()
            .filter(|r| r.suite == "scaling_d64")
            .map(|r| r.seq)
            .collect();
        seqs.sort_unstable();
        seqs.dedup();
        for seq in seqs {
            let get = |name: &str| -> String {
                self.rows
                    .iter()
                    .find(|r| r.suite == "scaling_d64" && r.seq == seq && r.implementation == name)
                    .map(|r| {
                        if r.latency_ms.is_nan() {
                            "skipped".to_string()
                        } else {
                            format!("{:.3} ms", r.latency_ms)
                        }
                    })
                    .unwrap_or_else(|| "-".to_string())
            };
            println!(
                "{seq:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
                get("cpu_flash_1t"),
                get("cpu_parallel"),
                get("gpu_flash"),
                get("gpu_naive")
            );
        }
    }
}

fn cpu_iterations(flops: f64, rate_flops_per_s: f64, target_seconds: f64) -> usize {
    if flops <= 0.0 {
        return 1;
    }
    let per_call = flops / rate_flops_per_s;
    let n = (target_seconds / per_call).ceil() as usize;
    n.clamp(1, 12)
}

/// Run `f` once to warm up, then `iterations` timed runs. Returns
/// `(median_ms, min_ms, last_output)`.
fn time_cpu<F>(mut f: F, iterations: usize) -> (f64, f64, Tensor4)
where
    F: FnMut() -> Tensor4,
{
    let mut out = f();
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        out = f();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = median(&times);
    (median, *times.first().unwrap(), out)
}

fn median(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn error_stats(got: &Tensor4, want: &Tensor4) -> (f64, f64, f64) {
    assert_eq!(got.data.len(), want.data.len());
    if got.data.len() != want.data.len() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (a, b) in got.data.iter().zip(&want.data) {
        let a = *a as f64;
        let b = *b as f64;
        let d = (a - b).abs();
        max = max.max(d);
        sum += d;
        diff_sq += d * d;
        ref_sq += b * b;
    }
    let n = got.data.len().max(1) as f64;
    (max, sum / n, (diff_sq / ref_sq.max(1e-30)).sqrt())
}

fn write_rows(path: &Path, rows: &[&Row]) {
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot write {}: {e}", path.display());
            return;
        }
    };
    let mut w = BufWriter::new(file);
    let _ = writeln!(
        w,
        "suite,implementation,backend,batch,heads,seq,dim,causal,latency_ms,throughput_ms,e2e_ms,tflops,max_abs_err,mean_abs_err,rel_l2,iterations,gpu_bytes,threads,block_size,query_block,note"
    );
    for r in rows {
        let _ = writeln!(
            w,
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{:.6},{:.6e},{:.6e},{:.6e},{},{},{},{},{},{}",
            r.suite,
            r.implementation,
            r.backend,
            r.batch,
            r.heads,
            r.seq,
            r.dim,
            r.causal,
            r.latency_ms,
            r.throughput_ms,
            r.e2e_ms
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(String::new),
            r.tflops,
            r.max_abs_err,
            r.mean_abs_err,
            r.rel_l2,
            r.iterations,
            r.gpu_bytes,
            r.threads,
            r.block_size,
            r.query_block,
            csv_escape(&r.note),
        );
    }
}

fn write_transfer_csv(path: &Path, rows: &[&Row]) {
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot write {}: {e}", path.display());
            return;
        }
    };
    let mut w = BufWriter::new(file);
    let _ = writeln!(
        w,
        "size_mib,upload_gbps,download_gbps,device_copy_gbps,iterations"
    );
    for r in rows {
        let _ = writeln!(
            w,
            "{:.0},{:.3},{:.3},{:.3},{}",
            r.gpu_bytes as f64 / (1u64 << 20) as f64,
            r.tflops,
            r.max_abs_err,
            r.mean_abs_err,
            r.iterations
        );
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
