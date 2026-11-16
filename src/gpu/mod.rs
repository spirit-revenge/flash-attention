//! macOS Metal GPU backend for Flash Attention.
//!
//! The module is intentionally self-contained: device/pipeline setup, device
//! buffers, dispatch, timing and validation all live here. The public surface
//! mirrors the CPU API (`flash_attention`, `naive_attention`) so the benchmark
//! harness can treat both backends uniformly.
//!
//! Design notes
//! ------------
//! * One 4- or 8-lane group owns one query row; the row's Q vector and the
//!   online output accumulator live in registers (`dim / lanes` floats per
//!   lane).
//! * A threadgroup stages a `[block_cols, dim]` K/V tile in threadgroup memory
//!   and reuses it for `query_block` queries.
//! * The online softmax state (`m`, `l`) is replicated across the query's lanes, so
//!   no cross-lane broadcast is needed after the dot-product reduction.
//! * The causal mask turns scores into `-inf`; it never diverges around a
//!   SIMD-group operation.
//! * Timings separate per-call latency, batched (launch-amortized) kernel time
//!   and end-to-end time including host/device copies, because on Apple
//!   silicon each of these has a very different cost profile.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::rc::Rc;
use std::time::Instant;

use metal::*;

use crate::attention::Tensor4;

/// MSL source compiled for every distinct head dimension.
const SHADER_SRC: &str = include_str!("attention.metal");

const MAX_HEAD_DIM: usize = 128;
/// Lanes of a warp assigned to one query row. Fewer lanes mean fewer
/// cross-lane reductions but more channels (and registers) per lane.
fn query_lanes(dim: usize) -> usize {
    if dim <= 64 { 4 } else { 8 }
}

/// Errors surfaced by the GPU backend. Everything is recoverable: the backend
/// never panics on user input.
#[derive(Debug)]
pub enum GpuError {
    /// No Metal device (e.g. headless CI, older hardware).
    NoDevice,
    /// Q/K/V shape mismatch or unsupported head dimension.
    Shape(String),
    /// Invalid block size / query block / scale.
    Options(String),
    /// A buffer would exceed `MTLDevice.maxBufferLength`.
    Memory(String),
    /// MSL compilation failed.
    Compile(String),
    /// Pipeline creation failed.
    Pipeline(String),
    /// Unsupported device / dispatch configuration.
    Unsupported(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "no Metal device available"),
            GpuError::Shape(msg) => write!(f, "invalid attention shape: {msg}"),
            GpuError::Options(msg) => write!(f, "invalid attention options: {msg}"),
            GpuError::Memory(msg) => write!(f, "GPU allocation failed: {msg}"),
            GpuError::Compile(msg) => write!(f, "Metal shader compilation failed: {msg}"),
            GpuError::Pipeline(msg) => write!(f, "Metal pipeline creation failed: {msg}"),
            GpuError::Unsupported(msg) => write!(f, "unsupported GPU configuration: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Tunables for a GPU attention call.
#[derive(Clone, Copy, Debug)]
pub struct AttnOptions {
    /// Softmax scale; defaults to `1 / sqrt(dim)`.
    pub scale: Option<f32>,
    /// Apply a lower-triangular causal mask.
    pub causal: bool,
    /// Number of keys staged per K/V tile (`block_cols`).
    pub block_size: usize,
    /// Number of queries per threadgroup (`block_rows`).
    pub query_block: usize,
}

impl Default for AttnOptions {
    fn default() -> Self {
        Self {
            scale: None,
            causal: false,
            block_size: 32,
            query_block: 32,
        }
    }
}

/// Static device capabilities, captured once at startup.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub unified_memory: bool,
    pub max_threadgroup_memory: u64,
    pub max_threads_per_threadgroup: u64,
    pub max_buffer_length: u64,
    pub recommended_max_working_set: u64,
    /// Wall-clock time to create the device/queue wrapper.
    pub startup_ms: f64,
}

/// Result of a device-resident GPU benchmark.
#[derive(Clone, Debug)]
pub struct GpuBench {
    /// Median latency when one dispatch is committed and waited per iteration
    /// (includes launch + synchronization overhead).
    pub per_call_ms: f64,
    /// Kernel-only time: `iterations` dispatches queued into one command
    /// buffer and divided by `iterations` (launch overhead amortized).
    pub amortized_ms: f64,
    /// Median time of upload + dispatch + download, buffers reused.
    pub end_to_end_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub iterations: usize,
    /// Bytes of the Q/K/V/O buffers (plus the score matrix for the naive path).
    pub gpu_bytes: u64,
    pub groups: [usize; 3],
    pub threads_per_group: usize,
    pub threadgroup_memory_bytes: u64,
}

/// Host/device copy bandwidth measurement.
#[derive(Clone, Debug)]
pub struct TransferBench {
    pub size_bytes: u64,
    pub upload_gbps: f64,
    pub download_gbps: f64,
    pub device_copy_gbps: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AttnParams {
    batch: u32,
    heads: u32,
    seq: u32,
    dim: u32,
    scale: f32,
    causal: u32,
    block_cols: u32,
    block_rows: u32,
}

/// Per-head-dimension compiled pipelines. MSL is specialized on
/// `LANE_DIM = dim / 8`, which lets the inner loops use vectorized
/// threadgroup-memory loads.
struct DimPipelines {
    dim: usize,
    flash: ComputePipelineState,
    scores: ComputePipelineState,
    softmax: ComputePipelineState,
    pv: ComputePipelineState,
    max_block_cols: usize,
    compile_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Flash,
    Naive,
}

struct LaunchCfg {
    groups: MTLSize,
    threads: MTLSize,
    groups_tuple: [usize; 3],
    threads_per_group: usize,
    threadgroup_memory_bytes: u64,
}

struct DeviceBuffers {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    out: Buffer,
    /// Materialized `[batch, heads, seq, seq]` scores for the naive path.
    scores: Option<Buffer>,
}

struct Prepared {
    kind: Kind,
    pipes: Rc<DimPipelines>,
    params: AttnParams,
    cfg: LaunchCfg,
    bufs: DeviceBuffers,
    gpu_bytes: u64,
}

/// A Metal-backed Flash Attention engine. Reuse one instance across calls so
/// pipelines and buffers can be re-used.
pub struct GpuAttention {
    device: Device,
    queue: CommandQueue,
    info: GpuInfo,
    pipelines: RefCell<HashMap<usize, Rc<DimPipelines>>>,
}

impl GpuAttention {
    /// Create the backend, or return [`GpuError::NoDevice`] when Metal is not
    /// available (the caller can then skip GPU benchmarks/tests gracefully).
    pub fn new() -> Result<Self, GpuError> {
        let t0 = Instant::now();
        let device = Device::system_default().ok_or(GpuError::NoDevice)?;
        let queue = device.new_command_queue();
        let info = GpuInfo {
            name: device.name().to_string(),
            unified_memory: device.has_unified_memory(),
            max_threadgroup_memory: device.max_threadgroup_memory_length() as u64,
            max_threads_per_threadgroup: device.max_threads_per_threadgroup().width as u64,
            max_buffer_length: device.max_buffer_length() as u64,
            recommended_max_working_set: device.recommended_max_working_set_size(),
            startup_ms: t0.elapsed().as_secs_f64() * 1e3,
        };
        Ok(Self {
            device,
            queue,
            info,
            pipelines: RefCell::new(HashMap::new()),
        })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Compile (if necessary) the pipelines for `dim` and report the one-time
    /// shader compilation cost in milliseconds.
    pub fn warmup_dim(&self, dim: usize) -> Result<f64, GpuError> {
        let pipes = self.pipelines(dim)?;
        Ok(pipes.compile_ms)
    }

    /// `(max_total_threads_per_threadgroup, thread_execution_width)` of the
    /// compiled flash pipeline — useful for occupancy diagnostics.
    pub fn flash_pipeline_limits(&self, dim: usize) -> Result<(u64, u64), GpuError> {
        let p = self.pipelines(dim)?;
        Ok((
            p.flash.max_total_threads_per_threadgroup(),
            p.flash.thread_execution_width(),
        ))
    }

    /// Largest legal number of keys per K/V tile for this head dimension.
    pub fn max_block_size(&self, dim: usize) -> usize {
        let by_memory = (self.info.max_threadgroup_memory / (2 * 4 * dim as u64)) as usize;
        by_memory.clamp(16, 128) & !15
    }

    /// End-to-end Flash Attention (allocation + upload + dispatch + readback).
    pub fn flash_attention(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
    ) -> Result<Tensor4, GpuError> {
        self.run_once_api(q, k, v, opts, Kind::Flash)
    }

    /// End-to-end naive attention with a materialized `seq x seq` score matrix.
    pub fn naive_attention(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
    ) -> Result<Tensor4, GpuError> {
        self.run_once_api(q, k, v, opts, Kind::Naive)
    }

    fn run_once_api(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
        kind: Kind,
    ) -> Result<Tensor4, GpuError> {
        objc::rc::autoreleasepool(|| {
            let prepared = self.prepare(q, k, v, opts, kind)?;
            self.upload(&prepared, q, k, v);
            self.run_kind(&prepared, 1);
            Ok(self.download(&prepared))
        })
    }

    /// Benchmark the Flash Attention path without per-call allocation.
    pub fn benchmark_flash(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
        iterations: usize,
    ) -> Result<GpuBench, GpuError> {
        self.benchmark(q, k, v, opts, Kind::Flash, iterations)
    }

    /// Benchmark the materialized-score path without per-call allocation.
    pub fn benchmark_naive(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
        iterations: usize,
    ) -> Result<GpuBench, GpuError> {
        self.benchmark(q, k, v, opts, Kind::Naive, iterations)
    }

    fn benchmark(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
        kind: Kind,
        iterations: usize,
    ) -> Result<GpuBench, GpuError> {
        if iterations == 0 {
            return Err(GpuError::Options("iterations must be > 0".into()));
        }
        objc::rc::autoreleasepool(|| {
            let prepared = self.prepare(q, k, v, opts, kind)?;
            self.upload(&prepared, q, k, v);

            // Warm up clocks, caches and the driver's shader cache. The
            // fanless test machine ramps its GPU clock over a few hundred
            // milliseconds, so give it a bit more than the bare minimum.
            let warmups = if kind == Kind::Flash { 6 } else { 2 };
            for _ in 0..warmups {
                self.run_kind(&prepared, 1);
            }

            let mut per_call = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let t0 = Instant::now();
                self.run_kind(&prepared, 1);
                per_call.push(t0.elapsed().as_secs_f64() * 1e3);
            }

            // All dispatches into a single command buffer: removes the
            // per-iteration CPU launch/sync cost and exposes kernel time.
            let t0 = Instant::now();
            self.run_kind(&prepared, iterations);
            let amortized_ms = t0.elapsed().as_secs_f64() * 1e3 / iterations as f64;

            // Upload + dispatch + download, the realistic single-call cost
            // when data starts (and ends) on the host.
            let mut e2e = Vec::with_capacity(iterations.min(25));
            for _ in 0..iterations.min(25) {
                let t0 = Instant::now();
                self.upload(&prepared, q, k, v);
                self.run_kind(&prepared, 1);
                std::hint::black_box(self.download(&prepared));
                e2e.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            e2e.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut sorted = per_call.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            Ok(GpuBench {
                per_call_ms: median(&sorted),
                amortized_ms,
                end_to_end_ms: median(&e2e),
                min_ms: *sorted.first().unwrap(),
                max_ms: *sorted.last().unwrap(),
                iterations,
                gpu_bytes: prepared.gpu_bytes,
                groups: prepared.cfg.groups_tuple,
                threads_per_group: prepared.cfg.threads_per_group,
                threadgroup_memory_bytes: prepared.cfg.threadgroup_memory_bytes,
            })
        })
    }

    /// Measure host→device, device→host and device→device copy bandwidth.
    pub fn transfer_benchmark(
        &self,
        size_bytes: u64,
        iterations: usize,
    ) -> Result<TransferBench, GpuError> {
        if size_bytes < 4 || iterations == 0 {
            return Err(GpuError::Options(
                "transfer size must be >= 4 bytes and iterations > 0".into(),
            ));
        }
        if size_bytes > self.info.max_buffer_length {
            return Err(GpuError::Memory(format!(
                "transfer buffer of {} bytes exceeds maxBufferLength {}",
                size_bytes, self.info.max_buffer_length
            )));
        }
        objc::rc::autoreleasepool(|| {
            let a = self
                .device
                .new_buffer(size_bytes, MTLResourceOptions::StorageModeShared);
            let b = self
                .device
                .new_buffer(size_bytes, MTLResourceOptions::StorageModeShared);
            let len = (size_bytes / 4) as usize;
            let host: Vec<f32> = (0..len).map(|i| (i % 17) as f32 * 0.25).collect();

            // Sustained bandwidth: batch the copies so the measurement is
            // not dominated by per-call launch/synchronization overhead.
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.as_ptr(),
                        a.contents() as *mut f32,
                        host.len(),
                    );
                }
            }
            let upload_ms = t0.elapsed().as_secs_f64() * 1e3 / iterations as f64;

            let mut sink = vec![0.0f32; len];
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        a.contents() as *const f32,
                        sink.as_mut_ptr(),
                        sink.len(),
                    );
                }
            }
            let download_ms = t0.elapsed().as_secs_f64() * 1e3 / iterations as f64;

            let t0 = Instant::now();
            let cb = self.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            for _ in 0..iterations {
                blit.copy_from_buffer(&a, 0, &b, 0, size_bytes);
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let device_ms = t0.elapsed().as_secs_f64() * 1e3 / iterations as f64;

            std::hint::black_box(&sink);
            let gbps = |ms: f64| (size_bytes as f64 / 1e9) / (ms / 1e3);

            Ok(TransferBench {
                size_bytes,
                upload_gbps: gbps(upload_ms),
                download_gbps: gbps(download_ms),
                device_copy_gbps: gbps(device_ms),
            })
        })
    }

    // -- internals ---------------------------------------------------------

    fn pipelines(&self, dim: usize) -> Result<Rc<DimPipelines>, GpuError> {
        if dim == 0 || dim % 8 != 0 || dim > MAX_HEAD_DIM {
            return Err(GpuError::Unsupported(format!(
                "head dim must be a multiple of 8 and <= {MAX_HEAD_DIM}, got {dim}"
            )));
        }
        if let Some(existing) = self.pipelines.borrow().get(&dim).cloned() {
            return Ok(existing);
        }

        let t0 = Instant::now();
        let lanes = query_lanes(dim);
        let lane_dim = dim / lanes;
        let source =
            format!("#define LANE_DIM {lane_dim}\n#define QUERY_LANES {lanes}\n{SHADER_SRC}");
        let library = self
            .device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(GpuError::Compile)?;

        let make = |name: &str| -> Result<ComputePipelineState, GpuError> {
            let function = library
                .get_function(name, None)
                .map_err(GpuError::Compile)?;
            self.device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(GpuError::Pipeline)
        };

        let pipes = DimPipelines {
            dim,
            flash: make("flash_attention_kernel")?,
            scores: make("scores_kernel")?,
            softmax: make("softmax_rows_kernel")?,
            pv: make("pv_kernel")?,
            max_block_cols: self.max_block_size(dim),
            compile_ms: t0.elapsed().as_secs_f64() * 1e3,
        };
        let pipes = Rc::new(pipes);
        self.pipelines.borrow_mut().insert(dim, Rc::clone(&pipes));
        Ok(pipes)
    }

    /// One-time compilation report: `(head_dim, compile_ms)` per dim used so far.
    pub fn compiled_dims(&self) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = self
            .pipelines
            .borrow()
            .values()
            .map(|p| (p.dim, p.compile_ms))
            .collect();
        v.sort_by_key(|(d, _)| *d);
        v
    }

    fn prepare(
        &self,
        q: &Tensor4,
        k: &Tensor4,
        v: &Tensor4,
        opts: AttnOptions,
        kind: Kind,
    ) -> Result<Prepared, GpuError> {
        let (b, h, s, d) = validate_tensors(q, k, v)?;
        let pipes = self.pipelines(d)?;

        let lanes = query_lanes(d);
        if kind == Kind::Flash
            && (opts.query_block == 0
                || (opts.query_block * lanes) % 32 != 0
                || opts.query_block * lanes > self.info.max_threads_per_threadgroup as usize)
        {
            return Err(GpuError::Options(format!(
                "query_block must satisfy query_block*{lanes} % 32 == 0 and query_block*{lanes} <= {}, got {}",
                self.info.max_threads_per_threadgroup, opts.query_block
            )));
        }

        let mut block_cols = opts.block_size;
        match kind {
            Kind::Flash => {
                if block_cols == 0 || block_cols % 16 != 0 {
                    return Err(GpuError::Options(format!(
                        "block_size must be a positive multiple of 16, got {block_cols}"
                    )));
                }
                if block_cols > pipes.max_block_cols {
                    return Err(GpuError::Options(format!(
                        "block_size {} exceeds the GPU limit {} for dim {}",
                        block_cols, pipes.max_block_cols, d
                    )));
                }
            }
            Kind::Naive => block_cols = 1,
        }

        let scale = opts.scale.unwrap_or_else(|| 1.0 / (d as f32).sqrt());
        if !scale.is_finite() {
            return Err(GpuError::Options("scale must be finite".into()));
        }

        let bh = b
            .checked_mul(h)
            .ok_or_else(|| GpuError::Shape("batch * heads overflow".into()))?;
        let elems = bh
            .checked_mul(s)
            .and_then(|x| x.checked_mul(d))
            .ok_or_else(|| GpuError::Shape("tensor size overflow".into()))?;
        if elems > u32::MAX as usize {
            return Err(GpuError::Unsupported(format!(
                "tensor has {elems} elements; uint32 addressing supports at most {}",
                u32::MAX
            )));
        }

        let params = AttnParams {
            batch: b as u32,
            heads: h as u32,
            seq: s as u32,
            dim: d as u32,
            scale,
            causal: u32::from(opts.causal),
            block_cols: block_cols as u32,
            block_rows: opts.query_block as u32,
        };

        let tensor_bytes = (elems * 4) as u64;
        let mut gpu_bytes = tensor_bytes * 4;
        if kind == Kind::Naive {
            let score_elems = bh
                .checked_mul(s)
                .and_then(|x| x.checked_mul(s))
                .ok_or_else(|| GpuError::Shape("score matrix size overflow".into()))?;
            let score_bytes = (score_elems as u64) * 4;
            if score_bytes > self.info.max_buffer_length {
                return Err(GpuError::Memory(format!(
                    "naive score matrix needs {:.2} GiB but maxBufferLength is {:.2} GiB",
                    score_bytes as f64 / (1u64 << 30) as f64,
                    self.info.max_buffer_length as f64 / (1u64 << 30) as f64
                )));
            }
            gpu_bytes += score_bytes;
        }

        let shared = match kind {
            Kind::Flash => 2 * (block_cols * d * 4) as u64,
            Kind::Naive => 0,
        };
        if shared > self.info.max_threadgroup_memory {
            return Err(GpuError::Unsupported(format!(
                "threadgroup tile needs {shared} bytes but the device allows {}",
                self.info.max_threadgroup_memory
            )));
        }

        let new_buf = |bytes: u64, what: &str| -> Result<Buffer, GpuError> {
            if bytes > self.info.max_buffer_length {
                return Err(GpuError::Memory(format!(
                    "{what} buffer of {bytes} bytes exceeds maxBufferLength {}",
                    self.info.max_buffer_length
                )));
            }
            Ok(self
                .device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
        };

        let q_buf = new_buf(tensor_bytes, "Q")?;
        let k_buf = new_buf(tensor_bytes, "K")?;
        let v_buf = new_buf(tensor_bytes, "V")?;
        let out_buf = new_buf(tensor_bytes, "output")?;
        let score_buf = if kind == Kind::Naive {
            Some(new_buf((bh as u64) * (s as u64) * (s as u64) * 4, "score")?)
        } else {
            None
        };

        let (groups, threads, threads_per_group, groups_tuple) = match kind {
            Kind::Flash => {
                let groups = MTLSize::new(div_ceil(s, opts.query_block) as u64, bh as u64, 1);
                let threads = MTLSize::new((opts.query_block * lanes) as u64, 1, 1);
                (
                    groups,
                    threads,
                    opts.query_block * lanes,
                    [div_ceil(s, opts.query_block), bh, 1],
                )
            }
            Kind::Naive => {
                let groups = MTLSize::new(s as u64, s as u64, bh as u64);
                let threads = MTLSize::new(16, 16, 1);
                (groups, threads, 256, [s, s, bh])
            }
        };

        Ok(Prepared {
            kind,
            pipes,
            params,
            cfg: LaunchCfg {
                groups,
                threads,
                groups_tuple,
                threads_per_group,
                threadgroup_memory_bytes: shared,
            },
            bufs: DeviceBuffers {
                q: q_buf,
                k: k_buf,
                v: v_buf,
                out: out_buf,
                scores: score_buf,
            },
            gpu_bytes,
        })
    }

    fn upload(&self, prepared: &Prepared, q: &Tensor4, k: &Tensor4, v: &Tensor4) {
        // The buffers were allocated with the exact tensor length and shared
        // storage, so a straight host memcpy is the upload.
        unsafe {
            copy_into(&prepared.bufs.q, &q.data);
            copy_into(&prepared.bufs.k, &k.data);
            copy_into(&prepared.bufs.v, &v.data);
        }
        std::hint::black_box(&prepared.bufs);
    }

    fn download(&self, prepared: &Prepared) -> Tensor4 {
        let p = &prepared.params;
        let len = p.batch as usize * p.heads as usize * p.seq as usize * p.dim as usize;
        let mut data = vec![0.0f32; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                prepared.bufs.out.contents() as *const f32,
                data.as_mut_ptr(),
                len,
            );
        }
        Tensor4::from_vec(
            p.batch as usize,
            p.heads as usize,
            p.seq as usize,
            p.dim as usize,
            data,
        )
    }

    /// Encode `repeats` full attention passes and wait for completion.
    fn run_kind(&self, prepared: &Prepared, repeats: usize) {
        let cb = self.queue.new_command_buffer();
        match prepared.kind {
            Kind::Flash => {
                let enc = cb.new_compute_command_encoder();
                self.bind_flash(&enc, prepared);
                for _ in 0..repeats {
                    enc.dispatch_thread_groups(prepared.cfg.groups, prepared.cfg.threads);
                }
                enc.end_encoding();
            }
            Kind::Naive => {
                for _ in 0..repeats {
                    let enc = cb.new_compute_command_encoder();
                    self.bind_scores(&enc, prepared);
                    enc.dispatch_thread_groups(prepared.cfg.groups, prepared.cfg.threads);
                    enc.end_encoding();

                    let total_rows = prepared.params.batch as u64
                        * prepared.params.heads as u64
                        * prepared.params.seq as u64;
                    let enc = cb.new_compute_command_encoder();
                    self.bind_softmax(&enc, prepared);
                    enc.dispatch_thread_groups(
                        MTLSize::new(div_ceil(total_rows as usize, 64) as u64, 1, 1),
                        MTLSize::new(64, 1, 1),
                    );
                    enc.end_encoding();

                    let enc = cb.new_compute_command_encoder();
                    self.bind_pv(&enc, prepared);
                    enc.dispatch_thread_groups(
                        MTLSize::new(
                            div_ceil(prepared.params.dim as usize, 16) as u64,
                            div_ceil(prepared.params.seq as usize, 16) as u64,
                            (prepared.params.batch * prepared.params.heads) as u64,
                        ),
                        MTLSize::new(16, 16, 1),
                    );
                    enc.end_encoding();
                }
            }
        }
        cb.commit();
        cb.wait_until_completed();
    }

    fn bind_flash(&self, enc: &ComputeCommandEncoderRef, prepared: &Prepared) {
        enc.set_compute_pipeline_state(&prepared.pipes.flash);
        enc.set_buffer(0, Some(&prepared.bufs.q), 0);
        enc.set_buffer(1, Some(&prepared.bufs.k), 0);
        enc.set_buffer(2, Some(&prepared.bufs.v), 0);
        enc.set_buffer(3, Some(&prepared.bufs.out), 0);
        set_params(enc, 4, &prepared.params);
        if prepared.cfg.threadgroup_memory_bytes > 0 {
            enc.set_threadgroup_memory_length(0, prepared.cfg.threadgroup_memory_bytes);
        }
    }

    fn bind_scores(&self, enc: &ComputeCommandEncoderRef, prepared: &Prepared) {
        enc.set_compute_pipeline_state(&prepared.pipes.scores);
        enc.set_buffer(0, Some(&prepared.bufs.q), 0);
        enc.set_buffer(1, Some(&prepared.bufs.k), 0);
        enc.set_buffer(2, Some(prepared.bufs.scores.as_ref().unwrap()), 0);
        set_params(enc, 3, &prepared.params);
    }

    fn bind_softmax(&self, enc: &ComputeCommandEncoderRef, prepared: &Prepared) {
        enc.set_compute_pipeline_state(&prepared.pipes.softmax);
        enc.set_buffer(0, Some(prepared.bufs.scores.as_ref().unwrap()), 0);
        set_params(enc, 1, &prepared.params);
    }

    fn bind_pv(&self, enc: &ComputeCommandEncoderRef, prepared: &Prepared) {
        enc.set_compute_pipeline_state(&prepared.pipes.pv);
        enc.set_buffer(0, Some(prepared.bufs.scores.as_ref().unwrap()), 0);
        enc.set_buffer(1, Some(&prepared.bufs.v), 0);
        enc.set_buffer(2, Some(&prepared.bufs.out), 0);
        set_params(enc, 3, &prepared.params);
    }
}

fn set_params(enc: &ComputeCommandEncoderRef, index: u64, params: &AttnParams) {
    enc.set_bytes(
        index,
        std::mem::size_of::<AttnParams>() as u64,
        params as *const AttnParams as *const c_void,
    );
}

unsafe fn copy_into(dst: &Buffer, src: &[f32]) {
    let needed = (src.len() * 4) as u64;
    assert!(
        dst.length() >= needed,
        "GPU buffer is smaller than the host tensor"
    );
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.contents() as *mut f32, src.len());
    }
}

fn validate_tensors(
    q: &Tensor4,
    k: &Tensor4,
    v: &Tensor4,
) -> Result<(usize, usize, usize, usize), GpuError> {
    if q.batch != k.batch
        || q.batch != v.batch
        || q.heads != k.heads
        || q.heads != v.heads
        || q.seq != k.seq
        || q.seq != v.seq
        || q.dim != k.dim
        || q.dim != v.dim
    {
        return Err(GpuError::Shape(format!(
            "q/k/v shapes differ: q={:?} k={:?} v={:?}",
            (q.batch, q.heads, q.seq, q.dim),
            (k.batch, k.heads, k.seq, k.dim),
            (v.batch, v.heads, v.seq, v.dim)
        )));
    }
    if q.batch == 0 || q.heads == 0 || q.seq == 0 || q.dim == 0 {
        return Err(GpuError::Shape(
            "batch, heads, seq and dim must all be non-zero".into(),
        ));
    }
    if q.dim % 8 != 0 {
        return Err(GpuError::Unsupported(format!(
            "head dim must be a multiple of 8 for the GPU kernel, got {}",
            q.dim
        )));
    }
    if q.dim > MAX_HEAD_DIM {
        return Err(GpuError::Unsupported(format!(
            "head dim {} exceeds the supported maximum {MAX_HEAD_DIM}",
            q.dim
        )));
    }
    let expected = q
        .batch
        .checked_mul(q.heads)
        .and_then(|x| x.checked_mul(q.seq))
        .and_then(|x| x.checked_mul(q.dim))
        .ok_or_else(|| GpuError::Shape("tensor size overflow".into()))?;
    if q.data.len() != expected || k.data.len() != expected || v.data.len() != expected {
        return Err(GpuError::Shape(format!(
            "tensor data lengths do not match shape {:?}",
            (q.batch, q.heads, q.seq, q.dim)
        )));
    }
    Ok((q.batch, q.heads, q.seq, q.dim))
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
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

/// Read a shared-storage buffer back as an owned Vec. Used by tests.
#[cfg(test)]
fn read_buffer(buf: &Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{
        make_qkv_example, naive_attention, naive_attention_causal, scaled_dot_product_attention,
    };

    fn max_abs_diff(a: &Tensor4, b: &Tensor4) -> f32 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn gpu() -> Option<GpuAttention> {
        match GpuAttention::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn flash_matches_naive_reference() {
        let Some(gpu) = gpu() else { return };
        for &(b, h, s, d) in &[
            (1, 1, 8, 8),
            (1, 2, 33, 16),
            (2, 3, 17, 64),
            (1, 1, 64, 128),
        ] {
            let (q, k, v) = make_qkv_example(b, h, s, d, 7);
            for &causal in &[false, true] {
                for &bc in &[16usize, 32] {
                    let opts = AttnOptions {
                        causal,
                        block_size: bc.min(gpu.max_block_size(d)).max(16),
                        query_block: 8,
                        ..Default::default()
                    };
                    let got = gpu.flash_attention(&q, &k, &v, opts).unwrap();
                    let want = if causal {
                        naive_attention_causal(&q, &k, &v)
                    } else {
                        naive_attention(&q, &k, &v)
                    };
                    let diff = max_abs_diff(&got, &want);
                    assert!(
                        diff < 2e-4,
                        "gpu flash mismatch: shape {:?} causal={causal} bc={bc} diff={diff}",
                        (b, h, s, d)
                    );
                }
            }
        }
    }

    #[test]
    fn naive_gpu_matches_naive_reference() {
        let Some(gpu) = gpu() else { return };
        for &(b, h, s, d) in &[(1, 1, 9, 8), (2, 2, 20, 32)] {
            let (q, k, v) = make_qkv_example(b, h, s, d, 11);
            for &causal in &[false, true] {
                let got = gpu
                    .naive_attention(
                        &q,
                        &k,
                        &v,
                        AttnOptions {
                            causal,
                            query_block: 8,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let want = if causal {
                    naive_attention_causal(&q, &k, &v)
                } else {
                    naive_attention(&q, &k, &v)
                };
                let diff = max_abs_diff(&got, &want);
                assert!(
                    diff < 2e-4,
                    "gpu naive mismatch: shape {:?} causal={causal} diff={diff}",
                    (b, h, s, d)
                );
            }
        }
    }

    #[test]
    fn long_odd_shapes_and_large_scales_match_reference() {
        let Some(gpu) = gpu() else { return };
        let (q, k, v) = make_qkv_example(1, 2, 1000, 64, 77);
        for &causal in &[false, true] {
            for &scale in &[None, Some(8.0f32)] {
                let opts = AttnOptions {
                    causal,
                    scale,
                    block_size: 32,
                    query_block: 32,
                };
                let got = gpu.flash_attention(&q, &k, &v, opts).unwrap();
                let want = scaled_dot_product_attention(&q, &k, &v, scale, causal, 64);
                let diff = max_abs_diff(&got, &want);
                assert!(
                    diff < 5e-4,
                    "long-shape mismatch: causal={causal} scale={scale:?} diff={diff}"
                );
            }
        }
    }

    #[test]
    fn invalid_options_are_rejected() {
        let Some(gpu) = gpu() else { return };
        let (q, k, v) = make_qkv_example(1, 1, 8, 64, 3);
        let bad_dim = Tensor4::from_vec(1, 1, 8, 12, vec![0.0; 8 * 12 * 1]);
        assert!(
            gpu.flash_attention(&bad_dim, &bad_dim, &bad_dim, AttnOptions::default())
                .is_err()
        );
        let bad_block = AttnOptions {
            block_size: 24,
            ..Default::default()
        };
        assert!(gpu.flash_attention(&q, &k, &v, bad_block).is_err());
        let bad_query = AttnOptions {
            query_block: 6,
            ..Default::default()
        };
        assert!(gpu.flash_attention(&q, &k, &v, bad_query).is_err());
    }

    #[test]
    fn shared_buffer_roundtrip() {
        let Some(gpu) = gpu() else { return };
        let buf = gpu
            .device
            .new_buffer(64, MTLResourceOptions::StorageModeShared);
        let host: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        unsafe { copy_into(&buf, &host) };
        assert_eq!(read_buffer(&buf, 16), host);
    }
}
