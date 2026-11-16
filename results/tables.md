# Generated comparison tables

## Environment

| item | value |
|---|---|
| GPU | Apple M4 |
| GPU unified memory | true |
| threadgroup memory | 32768 bytes |
| max buffer | 8.88 GiB |
| CPU threads | 10 |
| Rust | rustc 1.99.0 (b940084d7 2026-09-28) |

## Startup and one-time costs

| item | ms | note |
|---|---|---|
| device_and_queue | 30.08 | GpuAttention::new (device + command queue) |
| shader_compile_d64 | 71.65 | MSL compile + 4 compute pipelines |
| shader_compile_d128 | 40.90 | MSL compile + 4 compute pipelines |
| first_call_d64 | 41.11 | includes MSL compile + pipeline creation |
| second_call_d64 | 0.78 | warm end-to-end |

## Sequence-length scaling, D=64, B*H=8 (non-causal)

Latency in milliseconds (GPU flash = median per-call latency; CPU = median call).

| S | CPU naive 1T | CPU flash 1T | CPU cached 1T | CPU all cores | GPU flash | GPU naive |
|---|---|---|---|---|---|---|
| 128 | 5.814 | 3.747 | 2.398 | 0.660 | 0.538 | 2.187 |
| 256 | 21.866 | 13.921 | 9.868 | 2.617 | 1.139 | 4.667 |
| 512 | 109.941 | 55.550 | 39.140 | 10.497 | 2.322 | 9.093 |
| 1024 | 458.465 | 226.561 | 156.149 | 34.670 | 3.145 | 33.140 |
| 2048 | 1797.272 | 895.250 | 627.862 | 142.388 | 11.952 | 133.593 |
| 4096 | 7206.298 | 3570.255 | 2589.609 | 576.490 | 47.826 | 537.544 |

### Speedup and GPU efficiency

| S | GPU/CPU 1T | GPU/CPU all cores | GPU flash / GPU naive | GPU flash TFLOP/s | max abs err |
|---|---|---|---|---|---|
| 128 | 7.0× | 1.2× | 4.1× | 0.12 | 5.36e-07 |
| 256 | 12.2× | 2.3× | 4.1× | 0.22 | 1.07e-06 |
| 512 | 23.9× | 4.5× | 3.9× | 0.36 | 1.37e-06 |
| 1024 | 72.0× | 11.0× | 10.5× | 0.74 | 2.03e-06 |
| 2048 | 74.9× | 11.9× | 11.2× | 0.73 | 2.89e-06 |
| 4096 | 74.7× | 12.1× | 11.2× | 0.72 | 3.93e-06 |

## Sequence-length scaling, D=128, B*H=8 (non-causal)

| S | CPU naive 1T | CPU flash 1T | CPU cached 1T | CPU all cores | GPU flash | GPU naive |
|---|---|---|---|---|---|---|
| 256 | 45.820 | 24.027 | 19.003 | 4.570 | 1.077 | 3.948 |
| 512 | 227.620 | 99.282 | 79.283 | 19.598 | 2.026 | 10.139 |
| 1024 | 859.067 | 385.948 | 312.707 | 77.675 | 6.600 | 41.160 |
| 2048 | 3497.251 | 1545.696 | 1249.162 | 259.069 | 25.910 | 167.163 |

| S | GPU/CPU 1T | GPU/CPU all cores | GPU flash / GPU naive | GPU flash TFLOP/s | max abs err |
|---|---|---|---|---|---|
| 256 | 22.3× | 4.2× | 3.7× | 0.37 | 8.94e-07 |
| 512 | 49.0× | 9.7× | 5.0× | 0.68 | 1.37e-06 |
| 1024 | 58.5× | 11.8× | 6.2× | 0.69 | 2.15e-06 |
| 2048 | 59.7× | 10.0× | 6.5× | 0.67 | 2.68e-06 |

## Causal masking, D=64, B*H=8

| S | CPU 1T (ms) | CPU all cores (ms) | GPU flash (ms) | GPU naive (ms) | GPU/CPU all cores | max abs err |
|---|---|---|---|---|---|---|
| 1024 | 116.205 | 22.273 | 1.929 | 32.765 | 11.5× | 1.61e-06 |
| 2048 | 450.658 | 74.562 | 6.392 | 133.038 | 11.7× | 2.32e-06 |
| 4096 | 1797.818 | 300.406 | 24.349 | 536.085 | 12.3× | 4.11e-06 |

## Batch/head scaling (S=1024, D=64)

| B*H | CPU 1T (ms) | CPU all cores (ms) | GPU flash (ms) | GPU naive (ms) | GPU/CPU all cores |
|---|---|---|---|---|---|
| 1 | 28.258 | 4.583 | 1.225 | 6.001 | 3.7× |
| 4 | 113.191 | 18.580 | 1.786 | 16.636 | 10.4× |
| 8 | 230.911 | 35.366 | 3.047 | 32.736 | 11.6× |
| 16 | 455.717 | 71.240 | 5.970 | 65.996 | 11.9× |

## CPU thread scaling (S=1024, D=64, B*H=8)

| threads | flash ms | flash speedup | flash efficiency | naive ms | naive speedup |
|---|---|---|---|---|---|
| 1 | 167.099 | 1.00× | 100% | 478.856 | 1.00× |
| 2 | 95.823 | 1.74× | 87% | 269.605 | 1.78× |
| 4 | 56.206 | 2.97× | 74% | 157.421 | 3.04× |
| 6 | 47.321 | 3.53× | 59% | 126.854 | 3.77× |
| 8 | 40.496 | 4.13× | 52% | 102.850 | 4.66× |
| 10 | 37.559 | 4.45× | 44% | 90.959 | 5.26× |

## Block / tile-size sweep (S=1024, D=64)

| CPU block | CPU cached 1T (ms) | CPU all cores (ms) | GPU bc | GPU br | GPU flash (ms) | GPU err |
|---|---|---|---|---|---|---|
| 16 | 177.716 | 36.969 |  |  |  |  |
| 32 | 174.031 | 36.162 |  |  |  |  |
| 64 | 176.720 | 37.657 |  |  |  |  |
| 128 | 169.518 | 35.207 |  |  |  |  |
| 256 | 172.125 | 37.966 |  |  |  |  |
| 512 | 181.768 | 36.020 |  |  |  |  |
| 1024 | 173.641 | 35.958 |  |  |  |  |
|  |  |  | 16 | 16 | 3.238 | 1.97e-06 |
|  |  |  | 16 | 32 | 4.103 | 1.97e-06 |
|  |  |  | 16 | 64 | 3.571 | 1.97e-06 |
|  |  |  | 32 | 16 | 4.063 | 1.97e-06 |
|  |  |  | 32 | 32 | 2.825 | 1.97e-06 |
|  |  |  | 32 | 64 | 3.276 | 1.97e-06 |
|  |  |  | 48 | 16 | 6.248 | 1.97e-06 |
|  |  |  | 48 | 32 | 3.355 | 1.97e-06 |
|  |  |  | 48 | 64 | 3.263 | 1.97e-06 |
|  |  |  | 64 | 16 | 8.494 | 1.97e-06 |
|  |  |  | 64 | 32 | 4.478 | 1.97e-06 |
|  |  |  | 64 | 64 | 2.915 | 1.97e-06 |

GPU rows: `bc` = keys per staged K/V tile, `br` = queries per threadgroup.

## Host/device transfer bandwidth

| size (MiB) | host→device (GB/s) | device→host (GB/s) | device→device (GB/s) |
|---|---|---|---|
| 1 | 59.9 | 75.9 | 6.9 |
| 8 | 33.2 | 56.2 | 21.8 |
| 64 | 32.7 | 29.0 | 20.4 |
| 256 | 19.1 | 22.1 | 18.3 |

## Numerical accuracy (all benchmarked shapes)

| implementation | max abs error | max relative L2 | worst case |
|---|---|---|---|
| CPU flash, baseline (1 thread) | 3.755e-06 | 1.342e-06 | S=4096, D=64, causal=False |
| CPU flash, cached (1 thread) | 3.874e-06 | 9.213e-07 | S=4096, D=64, causal=True |
| CPU flash, parallel (all cores) | 3.874e-06 | 9.213e-07 | S=4096, D=64, causal=True |
| GPU flash (Metal) | 4.113e-06 | 9.338e-07 | S=4096, D=64, causal=True |
| GPU naive (S² materialized) | 3.219e-06 | 1.268e-06 | S=4096, D=64, causal=True |

Errors are computed against the original single-threaded CPU naive implementation.

## GPU memory footprint (B*H=8, D=64)

| S | naive scores (MiB) | flash Q/K/V/O (MiB) | ratio naive/flash |
|---|---|---|---|
| 512 | 8.0 | 4.0 | 2.0× |
| 1024 | 32.0 | 8.0 | 4.0× |
| 2048 | 128.0 | 16.0 | 8.0× |
| 4096 | 512.0 | 32.0 | 16.0× |
| 8192 | 2048.0 | 64.0 | 32.0× |
