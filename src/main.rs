use flash_attention::{Tensor4, flash_attention_cpu, make_qkv_example, naive_attention};
use std::time::Instant;

fn max_abs_diff(a: &Tensor4, b: &Tensor4) -> f32 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A small end-to-end demo: run the reference, the CPU Flash Attention and (if
/// the `metal` feature is enabled) the GPU Flash Attention, then compare
/// outputs and timings.
fn main() {
    // Head dims used by real models (64/128) are also the dims supported by
    // the GPU kernel.
    let (q, k, v) = make_qkv_example(1, 8, 512, 64, 1234);

    println!(
        "batch={}, heads={}, seq={}, dim={}",
        q.batch, q.heads, q.seq, q.dim
    );

    let t0 = Instant::now();
    let ref_out = naive_attention(&q, &k, &v);
    let ref_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let cpu_out = flash_attention_cpu(&q, &k, &v, 64);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!("reference (naive CPU): {ref_ms:9.3} ms");
    println!("flash attention (CPU): {cpu_ms:9.3} ms");
    println!(
        "CPU max abs error vs reference: {:.3e}",
        max_abs_diff(&cpu_out, &ref_out)
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use flash_attention::gpu::{AttnOptions, GpuAttention};
        match GpuAttention::new() {
            Ok(gpu) => {
                let opts = AttnOptions {
                    block_size: 32,
                    query_block: 32,
                    ..Default::default()
                };
                // Warm-up (pipeline compilation + caches) is not timed.
                let _ = gpu.flash_attention(&q, &k, &v, opts);

                let t0 = Instant::now();
                let gpu_out = gpu.flash_attention(&q, &k, &v, opts).unwrap();
                let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;

                println!(
                    "flash attention (GPU): {gpu_ms:9.3} ms  [{}]",
                    gpu.info().name
                );
                println!(
                    "GPU max abs error vs reference: {:.3e}",
                    max_abs_diff(&gpu_out, &ref_out)
                );
                println!(
                    "speedup vs naive CPU: {:.1}x, vs CPU flash: {:.1}x",
                    ref_ms / gpu_ms,
                    cpu_ms / gpu_ms
                );
            }
            Err(e) => println!("GPU backend unavailable: {e}"),
        }
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("GPU demo disabled; rebuild with `--features metal` on macOS.");
}
