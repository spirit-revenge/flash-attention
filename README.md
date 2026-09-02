# Flash Attention

[中文](README_CN.md) | English

A CPU implementation of Flash Attention and Scaled Dot-Product Attention in Rust, with optional Python and NumPy support.

## Features

- Pure Rust CPU implementation
- Block-wise attention with online softmax
- Multi-batch and multi-head inputs
- Optional causal masking
- Configurable scale and block size
- Optional PyO3 Python extension

## Installation

### Rust

Build and test the Rust library:

```bash
cargo build --release
cargo test
```

### Python

Install the build dependencies:

```bash
python3 -m pip install numpy maturin
```

Build and install the Python extension from the project root:

```bash
maturin build --release --features python
python3 -m pip install --force-reinstall target/wheels/*.whl
```

## Usage

### Python

Inputs use the shape `[batch, heads, seq, dim]`. Lower-rank inputs such as `[seq, dim]` are also supported.

```python
import numpy as np
import flash_attention

q = np.random.randn(2, 3, 5, 4).astype(np.float32)
k = np.random.randn(2, 3, 5, 4).astype(np.float32)
v = np.random.randn(2, 3, 5, 4).astype(np.float32)

output = flash_attention.flash_attention_py(
    q,
    k,
    v,
    causal=False,
    block_size=2,
)

print(np.asarray(output).shape)  # (2, 3, 5, 4)
```

For standard SDPA naming, use:

```python
output = flash_attention.scaled_dot_product_attention_py(
    q, k, v, causal=True
)
```

Run the included smoke demo with:

```bash
python3 test.py
```

> [!WARNING]
> This is a personal learning project; please do not use it for formal projects.